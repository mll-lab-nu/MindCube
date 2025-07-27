"""
Qwen2.5-VL Inference Engine with support for both transformers and vLLM backends.
"""

import torch
from typing import List, Dict, Any, Optional
from PIL import Image
import traceback
import os

from ..base import BaseInferenceEngine
from ..utils import ModelLoader, ImageProcessor, ResponseProcessor, ConfigManager


class QwenInferenceEngine(BaseInferenceEngine):
    """Qwen2.5-VL Inference Engine with multiple backend support."""
    
    def __init__(self, model_path: str, backend: str = "transformers", **kwargs):
        """
        Initialize Qwen inference engine.
        
        Args:
            model_path: Path to the model
            backend: Inference backend ('transformers' or 'vllm')
            **kwargs: Additional configuration
        """
        super().__init__(model_path, "qwen2.5vl", **kwargs)
        self.backend = backend
        self.vllm_engine = None
        
        # Load default config and update with user config
        self.config.update(ConfigManager.get_default_config("qwen2.5vl"))
        self.config.update(kwargs)
        
        # Check backend compatibility
        compatibility = ModelLoader.check_model_compatibility(model_path)
        if backend == "vllm" and not compatibility["vllm"]:
            print("Warning: vLLM not available, falling back to transformers")
            self.backend = "transformers"
    
    def load_model(self) -> None:
        """Load the model and processor."""
        try:
            if self.backend == "vllm":
                self._load_vllm()
            else:
                self._load_transformers()
            print(f"Model loaded successfully using {self.backend} backend")
        except Exception as e:
            print(f"Error loading model: {e}")
            if self.backend == "vllm":
                print("Falling back to transformers backend")
                self.backend = "transformers"
                self._load_transformers()
            else:
                raise
    
    def _load_transformers(self) -> None:
        """Load model using transformers backend."""
        self.model, self.processor = ModelLoader.load_qwen_transformers(
            self.model_path,
            **self.config
        )
    
    def _load_vllm(self) -> None:
        """Load model using vLLM backend."""
        # Extract vllm_config from nested structure and merge with root config
        vllm_config = self.config.get('vllm_config', {})
        combined_config = {**self.config, **vllm_config}  # vllm_config overrides root config
        
        self.vllm_engine = ModelLoader.load_qwen_vllm(
            self.model_path,
            **combined_config
        )
        # For vLLM, we still need processor for tokenization
        from transformers.models.auto.processing_auto import AutoProcessor
        base_model = self.config.get('base_model', 'Qwen/Qwen2.5-VL-3B-Instruct')
        self.processor = AutoProcessor.from_pretrained(
            base_model,
            trust_remote_code=True,
            max_pixels=self.config.get('max_pixels', 480 * 480)
        )
    
    def process_input(self, prompt: str, image_paths: List[str], **kwargs) -> Any:
        """
        Process input data for inference.
        
        Args:
            prompt: Text prompt
            image_paths: List of image file paths
            **kwargs: Additional processing parameters
            
        Returns:
            Processed input ready for model inference
        """
        # Validate inputs
        if not self.validate_inputs(prompt, image_paths):
            raise ValueError("Invalid input data")
        
        # Load and validate images
        images, errors = ImageProcessor.load_and_validate_images(image_paths)
        if errors:
            print(f"Image loading errors: {errors}")
        
        if self.backend == "vllm":
            return self._process_input_vllm(prompt, images, **kwargs)
        else:
            return self._process_input_transformers(prompt, images, **kwargs)
    
    def _process_input_transformers(self, prompt: str, images: List[Image.Image], **kwargs) -> Dict:
        """Process input for transformers backend."""
        # Create messages
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }]
        
        # Add images to messages only if images exist
        if images:
            max_pixels = kwargs.get('max_pixels', self.config.get('max_pixels', 480 * 480))
            image_messages = ImageProcessor.prepare_image_messages(images, max_pixels)
            for img_msg in image_messages:
                messages[0]["content"].insert(0, img_msg)
        
        # Prepare text template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process input - handle empty images case
        if images:
            inputs = self.processor(
                text=[text],
                images=images,
                return_tensors="pt"
            )
        else:
            # For text-only input, don't pass images parameter
            inputs = self.processor(
                text=[text],
                return_tensors="pt"
            )
        
        # Move inputs to CUDA if available
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        return {
            'inputs': inputs,
            'images': images,
            'messages': messages
        }
    
    def _process_input_vllm(self, prompt: str, images: List[Image.Image], **kwargs) -> Dict:
        """Process input for vLLM backend."""
        # Create messages like transformers for consistency
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }]
        
        # Add images to messages if they exist
        if images:
            max_pixels = kwargs.get('max_pixels', self.config.get('max_pixels', 480 * 480))
            image_messages = ImageProcessor.prepare_image_messages(images, max_pixels)
            for img_msg in image_messages:
                messages[0]["content"].insert(0, img_msg)
        
        # Format prompt using the same template as transformers
        formatted_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        return {
            'prompt': formatted_prompt,  # Use formatted prompt instead of raw prompt
            'images': images,
            'messages': messages
        }
    
    def generate_response(self, processed_input: Any, **kwargs) -> str:
        """
        Generate response from processed input.
        
        Args:
            processed_input: Output from process_input
            **kwargs: Generation parameters
            
        Returns:
            Generated response text
        """
        try:
            if self.backend == "vllm":
                return self._generate_vllm(processed_input, **kwargs)
            else:
                return self._generate_transformers(processed_input, **kwargs)
        except Exception as e:
            print(f"Error in generate_response: {e}")
            print(traceback.format_exc())
            return f"Error: {str(e)}"
    
    def _generate_transformers(self, processed_input: Dict, **kwargs) -> str:
        """Generate response using transformers backend."""
        inputs = processed_input['inputs']
        
        # Generation parameters - filter out non-generation parameters
        generation_config = self.config.get('generation_config', {}).copy()
        
        # Only include valid generation parameters
        valid_gen_params = {
            'max_new_tokens', 'temperature', 'top_p', 'top_k', 'do_sample',
            'num_beams', 'repetition_penalty', 'length_penalty', 'early_stopping',
            'pad_token_id', 'eos_token_id', 'use_cache'
        }
        
        # Filter kwargs to only include valid generation parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_gen_params}
        generation_config.update(filtered_kwargs)
        generation_config.setdefault('max_new_tokens', self.config.get('max_new_tokens', 4096))
        
        # FORCE deterministic inference using greedy decoding
        generation_config['do_sample'] = False  # Forces greedy decoding (ignores all sampling params)
        generation_config['temperature'] = 0.0  # Backup for safety (ignored when do_sample=False)
        
        # Generate response
        generated_ids = self.model.generate(**inputs, **generation_config)
        
        # Process output
        if len(generated_ids) > 0 and len(inputs["input_ids"]) > 0:
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in 
                zip(inputs["input_ids"], generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            response = output_text[0] if output_text else "No response generated"
        else:
            response = "Error: No response generated"
        
        return ResponseProcessor.clean_response(response)
    
    def _generate_vllm(self, processed_input: Dict, **kwargs) -> str:
        """Generate response using vLLM backend with multimodal support."""
        try:
            from vllm import SamplingParams
            
            # FORCE deterministic inference - temperature=0.0 enables greedy sampling in vLLM
            sampling_params = SamplingParams(
                temperature=0.0,  # vLLM automatically enables greedy sampling and ignores other params
                max_tokens=min(kwargs.get('max_new_tokens', 1536), 1536),
                min_tokens=1,
                seed=42          # Fixed seed for additional reproducibility
            )
            
            # Use proper vLLM multimodal interface
            prompt = processed_input['prompt']
            images = processed_input.get('images', [])
            
            if images:
                # For vLLM with multimodal data, use the new format
                # Create input dict with both prompt and multi_modal_data
                llm_inputs = {
                    "prompt": prompt,
                    "multi_modal_data": {"image": images}
                }
                outputs = self.vllm_engine.generate([llm_inputs], sampling_params)
            else:
                # Text-only generation
                outputs = self.vllm_engine.generate([prompt], sampling_params)
            
            response = outputs[0].outputs[0].text if outputs else "No response generated"
            return ResponseProcessor.clean_response(response)
            
        except Exception as e:
            print(f"vLLM generation error: {e}")
            print(f"Falling back to text-only mode")
            # Fallback to text-only if multimodal fails
            try:
                prompt = processed_input['prompt']
                outputs = self.vllm_engine.generate([prompt], sampling_params)
                response = outputs[0].outputs[0].text if outputs else "No response generated"
                return ResponseProcessor.clean_response(response)
            except Exception as e2:
                return f"Error in vLLM generation: {str(e2)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_path': self.model_path,
            'model_type': self.model_type,
            'backend': self.backend,
            'config': self.config,
            'loaded': self.model is not None or self.vllm_engine is not None
        }
    
    def set_generation_config(self, **config) -> None:
        """Update generation configuration."""
        if 'generation_config' not in self.config:
            self.config['generation_config'] = {}
        self.config['generation_config'].update(config) 
    
    def process_batch(self, batch_data: List[Dict], image_root: str, **kwargs) -> List[Optional[Dict]]:
        """
        Optimized batch processing for Qwen models.
        
        Args:
            batch_data: List of data samples
            image_root: Root directory for images
            **kwargs: Additional parameters
            
        Returns:
            List of results (None for failed samples)
        """
        if self.backend == "vllm":
            # vLLM naturally supports batching
            return self._process_batch_vllm(batch_data, image_root, **kwargs)
        else:
            # For transformers, implement custom batching
            return self._process_batch_transformers(batch_data, image_root, **kwargs)
    
    def _process_batch_transformers(self, batch_data: List[Dict], image_root: str, **kwargs) -> List[Optional[Dict]]:
        """
        Batch processing for transformers backend.
        """
        # Load model if not already loaded (check both transformers and vLLM)
        if self.backend == "transformers" and self.model is None:
            self.load_model()
        elif self.backend == "vllm" and self.vllm_engine is None:
            self.load_model()
        
        # Prepare batch inputs
        batch_prompts = []
        batch_images_list = []
        valid_indices = []
        
        for idx, data in enumerate(batch_data):
            try:
                # Extract and validate data
                prompt = data.get('input_prompt', '')
                image_paths = data.get('images', [])
                
                if not prompt:
                    print(f"Warning: No input_prompt found for batch sample {idx}")
                    continue
                
                # Process image paths - FORCE all paths to use image_root
                if image_paths:
                    updated_image_paths = []
                    for img_path in image_paths:
                        # Extract just the relative part of the path for consistent mapping
                        if img_path.startswith(image_root):
                            # Already has correct prefix
                            updated_image_paths.append(img_path)
                        else:
                            # For ANY other path (relative, absolute, Windows, etc.), 
                            # extract the meaningful part and prepend image_root
                            if os.path.isabs(img_path) or (len(img_path) > 2 and img_path[1] == ':'):
                                # For absolute paths (Unix or Windows) like "D:/data/MindCube_image/other_all_image/...",
                                # extract the part after "MindCube_image/" or similar pattern
                                if "MindCube_image/" in img_path:
                                    relative_part = img_path.split("MindCube_image/", 1)[1]
                                elif "other_all_image/" in img_path:
                                    relative_part = "other_all_image/" + img_path.split("other_all_image/", 1)[1]
                                else:
                                    # Extract filename and assume it's in root
                                    relative_part = os.path.basename(img_path)
                            else:
                                # Relative path - use as is
                                relative_part = img_path
                            
                            # Always prepend image_root
                            updated_image_paths.append(os.path.join(image_root, relative_part))
                    image_paths = updated_image_paths
                
                # Load and validate images
                images, errors = ImageProcessor.load_and_validate_images(image_paths)
                if errors:
                    print(f"Image loading errors for batch sample {idx}: {errors}")
                
                batch_prompts.append(prompt)
                batch_images_list.append(images)
                valid_indices.append(idx)
                
            except Exception as e:
                print(f"Error preparing batch sample {idx}: {e}")
                continue
        
        if not batch_prompts:
            return [None] * len(batch_data)
        
        # Process batch
        try:
            batch_responses = self._generate_batch_transformers(batch_prompts, batch_images_list, **kwargs)
            
            # Prepare results
            results = [None] * len(batch_data)
            for i, (valid_idx, response) in enumerate(zip(valid_indices, batch_responses)):
                result = batch_data[valid_idx].copy()
                result['answer'] = response
                result['model_type'] = self.model_type
                results[valid_idx] = result
            
            return results
            
        except Exception as e:
            print(f"Error in batch processing: {e}")
            # Fallback to individual processing
            return super().process_batch(batch_data, image_root, **kwargs)
    
    def _generate_batch_transformers(self, batch_prompts: List[str], batch_images_list: List[List], **kwargs) -> List[str]:
        """
        Generate responses for a batch using transformers.
        """
        # Prepare batch inputs
        batch_texts = []
        all_images = []
        
        for prompt, images in zip(batch_prompts, batch_images_list):
            # Create messages
            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }]
            
            # Add images to messages
            max_pixels = kwargs.get('max_pixels', self.config.get('max_pixels', 480 * 480))
            image_messages = ImageProcessor.prepare_image_messages(images, max_pixels)
            for img_msg in image_messages:
                messages[0]["content"].insert(0, img_msg)
            
            # Prepare text template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            batch_texts.append(text)
            all_images.extend(images)  # Flatten all images
        
        # Process batch inputs
        inputs = self.processor(
            text=batch_texts,
            images=all_images if all_images else None,
            return_tensors="pt",
            padding=True  # Important for batch processing
        )
        
        # Move inputs to CUDA if available
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Generation parameters - filter out non-generation parameters
        generation_config = self.config.get('generation_config', {}).copy()
        
        # Only include valid generation parameters
        valid_gen_params = {
            'max_new_tokens', 'temperature', 'top_p', 'top_k', 'do_sample',
            'num_beams', 'repetition_penalty', 'length_penalty', 'early_stopping',
            'pad_token_id', 'eos_token_id', 'use_cache'
        }
        
        # Filter kwargs to only include valid generation parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_gen_params}
        generation_config.update(filtered_kwargs)
        generation_config.setdefault('max_new_tokens', self.config.get('max_new_tokens', 4096))
        
        # FORCE deterministic inference using greedy decoding
        generation_config['do_sample'] = False  # Forces greedy decoding (ignores all sampling params)
        generation_config['temperature'] = 0.0  # Backup for safety (ignored when do_sample=False)
        
        # Generate batch responses
        generated_ids = self.model.generate(**inputs, **generation_config)
        
        # Process batch outputs
        if len(generated_ids) > 0 and len(inputs["input_ids"]) > 0:
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in 
                zip(inputs["input_ids"], generated_ids)
            ]
            output_texts = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            # Clean responses
            batch_responses = []
            for output_text in output_texts:
                response = output_text if output_text else "No response generated"
                batch_responses.append(ResponseProcessor.clean_response(response))
            
            return batch_responses
        else:
            return ["Error: No response generated"] * len(batch_prompts)
    
    def _process_batch_vllm(self, batch_data: List[Dict], image_root: str, **kwargs) -> List[Optional[Dict]]:
        """
        Batch processing for vLLM backend with multimodal support.
        """
        # Load vLLM engine if not already loaded
        if self.vllm_engine is None:
            self.load_model()
        
        # Prepare batch inputs
        batch_prompts = []
        batch_images_list = []
        valid_indices = []
        
        for idx, data in enumerate(batch_data):
            try:
                # Extract and validate data
                prompt = data.get('input_prompt', '')
                image_paths = data.get('images', [])
                
                if not prompt:
                    print(f"Warning: No input_prompt found for batch sample {idx}")
                    continue
                
                # Process image paths - FORCE all paths to use image_root
                if image_paths:
                    updated_image_paths = []
                    for img_path in image_paths:
                        # Extract just the relative part of the path for consistent mapping
                        if img_path.startswith(image_root):
                            # Already has correct prefix
                            updated_image_paths.append(img_path)
                        else:
                            # For ANY other path (relative, absolute, Windows, etc.), 
                            # extract the meaningful part and prepend image_root
                            if os.path.isabs(img_path) or (len(img_path) > 2 and img_path[1] == ':'):
                                # For absolute paths (Unix or Windows) like "D:/data/MindCube_image/other_all_image/...",
                                # extract the part after "MindCube_image/" or similar pattern
                                if "MindCube_image/" in img_path:
                                    relative_part = img_path.split("MindCube_image/", 1)[1]
                                elif "other_all_image/" in img_path:
                                    relative_part = "other_all_image/" + img_path.split("other_all_image/", 1)[1]
                                else:
                                    # Extract filename and assume it's in root
                                    relative_part = os.path.basename(img_path)
                            else:
                                # Relative path - use as is
                                relative_part = img_path
                            
                            # Always prepend image_root
                            updated_image_paths.append(os.path.join(image_root, relative_part))
                    image_paths = updated_image_paths
                
                # Load and validate images
                images, errors = ImageProcessor.load_and_validate_images(image_paths)
                if errors:
                    print(f"Image loading errors for batch sample {idx}: {errors}")
                
                # Format prompt like in _process_input_vllm for consistency
                messages = [{
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }]
                
                # Add images to messages if they exist
                if images:
                    max_pixels = kwargs.get('max_pixels', self.config.get('max_pixels', 480 * 480))
                    image_messages = ImageProcessor.prepare_image_messages(images, max_pixels)
                    for img_msg in image_messages:
                        messages[0]["content"].insert(0, img_msg)
                
                # Format prompt using the same template as transformers
                formatted_prompt = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                batch_prompts.append(formatted_prompt)  # Use formatted prompt
                batch_images_list.append(images)
                valid_indices.append(idx)
                
            except Exception as e:
                print(f"Error preparing batch sample {idx}: {e}")
                continue
        
        if not batch_prompts:
            return [None] * len(batch_data)
        
        # Process batch with vLLM multimodal support
        try:
            from vllm import SamplingParams
            
            # FORCE deterministic inference - temperature=0.0 is vLLM equivalent of do_sample=False
            sampling_params = SamplingParams(
                temperature=0.0,  # Forces greedy decoding (deterministic)
                max_tokens=min(kwargs.get('max_new_tokens', 1536), 1536),  # Hard limit to 1536
                min_tokens=1,     # Ensure at least 1 token is generated
                seed=42          # Fixed seed for reproducibility
            )
            
            # Check if we have images in any sample
            has_images = any(len(images) > 0 for images in batch_images_list)
            
            if has_images:
                # For multimodal batch, we need to process individually since vLLM doesn't support multimodal batching yet
                print(f"Processing {len(batch_prompts)} samples with images individually...")
                batch_outputs = []
                
                for prompt, images in zip(batch_prompts, batch_images_list):
                    try:
                        if images:
                            # Use vLLM's multimodal data format with new API
                            llm_inputs = {
                                "prompt": prompt,
                                "multi_modal_data": {"image": images}
                            }
                            outputs = self.vllm_engine.generate([llm_inputs], sampling_params)
                        else:
                            # Text-only generation
                            outputs = self.vllm_engine.generate([prompt], sampling_params)
                        
                        batch_outputs.append(outputs[0] if outputs else None)
                        
                    except Exception as e:
                        print(f"Error processing individual sample with images: {e}")
                        # Fallback to text-only for this sample
                        try:
                            outputs = self.vllm_engine.generate([prompt], sampling_params)
                            batch_outputs.append(outputs[0] if outputs else None)
                        except Exception as e2:
                            print(f"Error in text-only fallback: {e2}")
                            batch_outputs.append(None)
            else:
                # Pure text batch - use vLLM's native batching
                batch_outputs = self.vllm_engine.generate(batch_prompts, sampling_params)
            
            # Prepare results
            results = [None] * len(batch_data)
            for i, (valid_idx, output) in enumerate(zip(valid_indices, batch_outputs)):
                if output and output.outputs:
                    response = output.outputs[0].text
                else:
                    response = "No response generated"
                
                result = batch_data[valid_idx].copy()
                result['answer'] = response
                result['model_type'] = self.model_type
                results[valid_idx] = result
            
            return results
            
        except Exception as e:
            print(f"Error in vLLM batch processing: {e}")
            # Fallback to individual processing only if batch fails
            return super().process_batch(batch_data, image_root, **kwargs) 