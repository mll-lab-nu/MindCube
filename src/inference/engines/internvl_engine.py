
import torch
from typing import List, Dict, Any, Optional
from PIL import Image
import traceback
import os
import torchvision.transforms as T

from ..base import BaseInferenceEngine
from ..utils import ModelLoader, ImageProcessor, ResponseProcessor, ConfigManager

# Constants for image preprocessing
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class InternVLInferenceEngine(BaseInferenceEngine):
    """InternVL Inference Engine with transformers backend."""
    
    def __init__(self, model_path: str, backend: str = "transformers", **kwargs):
        """
        Initialize InternVL inference engine.
        
        Args:
            model_path: Path to the model
            backend: Inference backend (only 'transformers' supported)
            **kwargs: Additional configuration
        """
        super().__init__(model_path, "internvl", **kwargs)
        self.backend = "transformers"  # Only transformers backend supported
        
        # Load default config and update with user config
        self.config.update(ConfigManager.get_default_config("internvl"))
        self.config.update(kwargs)
        
        # Set default values
        self.config.setdefault('input_size', 448)
        self.config.setdefault('max_num', 12)
        self.config.setdefault('use_thumbnail', True)
        self.config.setdefault('torch_dtype', torch.bfloat16)
        
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self) -> None:
        """Load the InternVL model and tokenizer."""
        try:
            from transformers.models.auto.tokenization_auto import AutoTokenizer
            from transformers.models.auto.modeling_auto import AutoModel
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=False
            )
            
            # Set pad_token_id to suppress warnings
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Load model
            self.model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=self.config.get('torch_dtype', torch.bfloat16),
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(self.device).eval()
            
            print(f"InternVL model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading InternVL model: {e}")
            raise
    
    def _build_transform(self, input_size: int):
        """Build image transform pipeline."""
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transform
    
    def _find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """Find the closest aspect ratio from target ratios."""
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio
    
    def _dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        """Dynamic preprocessing for images."""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        
        # Calculate target ratios
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) 
            for i in range(1, n + 1) for j in range(1, n + 1) 
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        
        # Find closest aspect ratio
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )
        
        # Calculate target dimensions
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        
        # Resize and split image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        
        # Add thumbnail if needed
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        
        return processed_images
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess a single image."""
        input_size = self.config.get('input_size', 448)
        max_num = self.config.get('max_num', 12)
        use_thumbnail = self.config.get('use_thumbnail', True)
        
        image = Image.open(image_path).convert('RGB')
        transform = self._build_transform(input_size)
        
        images = self._dynamic_preprocess(
            image, 
            image_size=input_size, 
            use_thumbnail=use_thumbnail, 
            max_num=max_num
        )
        
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)
        
        return pixel_values
    
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
        
        # Load images
        pixel_values_list = []
        for image_path in image_paths:
            try:
                pixel_values = self._load_image(image_path)
                pixel_values_list.append(pixel_values)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue
        
        # Concatenate all image tensors
        if pixel_values_list:
            pixel_values = torch.cat(pixel_values_list, dim=0)
            pixel_values = pixel_values.to(self.config.get('torch_dtype', torch.bfloat16)).to(self.device)
        else:
            pixel_values = None
        
        # Format prompt for multi-image input
        if len(image_paths) > 1:
            image_markers = [f"Image-{i+1}: <image>" for i in range(len(image_paths))]
            formatted_prompt = "\n".join(image_markers) + "\n" + prompt
        else:
            formatted_prompt = prompt
        
        return {
            'prompt': formatted_prompt,
            'pixel_values': pixel_values,
            'num_images': len(image_paths)
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
            if self.model is None or self.tokenizer is None:
                self.load_model()
            
            # Ensure model and tokenizer are loaded
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("Failed to load model or tokenizer")
            
            prompt = processed_input['prompt']
            pixel_values = processed_input['pixel_values']
            
            # Generation configuration - FORCE deterministic inference
            generation_config = {
                'max_new_tokens': kwargs.get('max_new_tokens', 500),
                'do_sample': False,  # FORCE greedy decoding for deterministic results
                'temperature': 0.0,  # Backup for safety (ignored when do_sample=False)
            }
            
            # Generate response using InternVL chat interface
            response = self.model.chat(
                self.tokenizer, 
                pixel_values, 
                prompt, 
                generation_config
            )
            
            return ResponseProcessor.clean_response(response)
            
        except Exception as e:
            print(f"Error in generate_response: {e}")
            print(traceback.format_exc())
            return f"Error: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_path': self.model_path,
            'model_type': self.model_type,
            'backend': self.backend,
            'config': self.config,
            'loaded': self.model is not None
        }
    
    def set_generation_config(self, **config) -> None:
        """Update generation configuration."""
        if 'generation_config' not in self.config:
            self.config['generation_config'] = {}
        self.config['generation_config'].update(config)
    
    def process_batch(self, batch_data: List[Dict], image_root: str, **kwargs) -> List[Optional[Dict]]:
        """
        Process a batch of data samples.
        
        Args:
            batch_data: List of data samples
            image_root: Root directory for images
            **kwargs: Additional parameters
            
        Returns:
            List of results (None for failed samples)
        """
        # Load model if not already loaded
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        results = []
        
        for idx, data in enumerate(batch_data):
            try:
                # Extract data
                prompt = data.get('input_prompt', '')
                image_paths = data.get('images', [])
                
                if not prompt:
                    print(f"Warning: No input_prompt found for batch sample {idx}")
                    results.append(None)
                    continue
                
                # Process image paths - ensure they use image_root
                if image_paths:
                    updated_image_paths = []
                    for img_path in image_paths:
                        if img_path.startswith(image_root):
                            updated_image_paths.append(img_path)
                        else:
                            # Handle path mapping like in Qwen engine
                            if os.path.isabs(img_path) or (len(img_path) > 2 and img_path[1] == ':'):
                                if "MindCube_image/" in img_path:
                                    relative_part = img_path.split("MindCube_image/", 1)[1]
                                elif "other_all_image/" in img_path:
                                    relative_part = "other_all_image/" + img_path.split("other_all_image/", 1)[1]
                                else:
                                    relative_part = os.path.basename(img_path)
                            else:
                                relative_part = img_path
                            updated_image_paths.append(os.path.join(image_root, relative_part))
                    image_paths = updated_image_paths
                
                # Process input
                processed_input = self.process_input(prompt, image_paths, **kwargs)
                
                # Generate response
                response = self.generate_response(processed_input, **kwargs)
                
                # Create result
                result = data.copy()
                result['answer'] = response
                result['model_type'] = self.model_type
                results.append(result)
                
            except Exception as e:
                print(f"Error processing batch sample {idx}: {e}")
                results.append(None)
        
        return results