"""
Utility classes and functions for inference.
"""

import os
import torch
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
import json
import re


class ModelLoader:
    """Utility class for loading different types of models."""
    
    @staticmethod
    def load_qwen_transformers(model_path: str, **kwargs):
        """
        Load Qwen2.5-VL model with transformers.
        
        Args:
            model_path: Path to the model
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (model, processor)
        """
        # Check if accelerate is available
        try:
            import accelerate
        except ImportError:
            print("Warning: accelerate not found. Installing...")
            import subprocess
            import sys
            subprocess.run([sys.executable, "-m", "pip", "install", "accelerate"], check=True)
            import accelerate
        
        from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
        from transformers.models.auto.processing_auto import AutoProcessor
        
        # Default parameters
        base_model = kwargs.get('base_model', 'Qwen/Qwen2.5-VL-3B-Instruct')
        max_pixels = kwargs.get('max_pixels', 480 * 480)
        torch_dtype = kwargs.get('torch_dtype', torch.float16)
        
        # GPU control parameters - simple and straightforward
        # Default to single GPU unless explicitly requested otherwise
        if kwargs.get('single_gpu', True):  # Default: single GPU
            device_map = {"": 0}  # Use only GPU 0
        else:
            device_map = 'auto'  # Multi-GPU auto mapping
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            base_model,
            trust_remote_code=True,
            max_pixels=max_pixels
        )
        
        # Prepare model loading arguments - keep it simple
        model_kwargs = {
            'torch_dtype': torch_dtype,
            'device_map': device_map,
            'trust_remote_code': True
        }
        
        # Load model
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        return model, processor
    
    @staticmethod
    def load_qwen_vllm(model_path: str, **kwargs):
        """
        Load Qwen2.5-VL model with vLLM for acceleration.
        Latest optimizations for memory management and stability.
        
        Args:
            model_path: Path to the model
            **kwargs: Additional parameters
            
        Returns:
            vLLM engine
        """
        try:
            from vllm import LLM, SamplingParams
            import gc
            
            # Clear any existing GPU memory first
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            
            # Get current GPU memory status
            available_memory_gb = 0
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                total_memory_gb = props.total_memory / (1024**3)
                allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
                available_memory_gb = total_memory_gb - allocated_gb
                
                print(f"GPU Memory Status:")
                print(f"  Total: {total_memory_gb:.2f} GB")
                print(f"  Allocated: {allocated_gb:.2f} GB") 
                print(f"  Available: {available_memory_gb:.2f} GB")
            
            # Use config parameters or sensible defaults
            tensor_parallel_size = kwargs.get('tensor_parallel_size', 1)
            
            # FIXED: Use config values properly with much larger defaults
            gpu_memory_utilization = kwargs.get('gpu_memory_utilization', 0.95)  # Use config value
            max_model_len = kwargs.get('max_model_len', 32768)  # Much larger for long inputs
            max_num_seqs = kwargs.get('max_num_seqs', 16)
            
            print(f"vLLM Configuration (from config file):")
            print(f"  GPU memory utilization: {gpu_memory_utilization}")
            print(f"  Max model length: {max_model_len}")
            print(f"  Max sequences: {max_num_seqs}")
            
            # Latest vLLM parameters for optimal performance and stability
            vllm_config = {
                'model': model_path,
                'tensor_parallel_size': tensor_parallel_size,
                'gpu_memory_utilization': gpu_memory_utilization,
                'max_model_len': max_model_len,
                'max_num_seqs': max_num_seqs,
                'trust_remote_code': True,
                'dtype': 'bfloat16',
                'enforce_eager': False,  # Enable CUDA graphs for performance
                'enable_chunked_prefill': True,  # Enable for throughput optimization
                'enable_prefix_caching': True,  # Enable for efficiency
                'block_size': 16,  # Standard block size
                'swap_space': 2,  # Conservative swap space
                'disable_custom_all_reduce': True,  # For single GPU stability
            }
            
            # Add max_num_batched_tokens only if explicitly provided in config
            if 'max_num_batched_tokens' in kwargs:
                vllm_config['max_num_batched_tokens'] = kwargs['max_num_batched_tokens']
                print(f"  Max batched tokens: {kwargs['max_num_batched_tokens']}")
            else:
                print(f"  Max batched tokens: Auto-managed by vLLM")
            
            # Add multimodal limits from config or use defaults
            limit_mm_per_prompt = kwargs.get('limit_mm_per_prompt', {"image": 4, "video": 1})
            try:
                vllm_config['limit_mm_per_prompt'] = limit_mm_per_prompt
            except:
                pass  # Older vLLM versions might not support this
            
            # Initialize vLLM engine with error handling
            print("Initializing vLLM engine...")
            llm = LLM(**vllm_config)
            print("✅ vLLM engine loaded successfully!")
            
            return llm
            
        except ImportError:
            raise ImportError("vLLM not installed. Please install vllm>=0.9.0 for latest optimizations.")
        except Exception as e:
            print(f"❌ Failed to load vLLM engine: {e}")
            print(f"Error type: {type(e).__name__}")
            
            # Cleanup on failure
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            
            print("💡 Suggestions:")
            print("  1. Try reducing gpu_memory_utilization to 0.3 or lower")
            print("  2. Close other GPU processes to free memory")
            print("  3. Use transformers backend as fallback")
            print("  4. Restart Python session to clear all GPU memory")
            
            raise
    
    @staticmethod
    def check_model_compatibility(model_path: str) -> Dict[str, bool]:
        """
        Check which inference backends are compatible with the model.
        
        Args:
            model_path: Path to the model
            
        Returns:
            Dictionary indicating backend compatibility
        """
        compatibility = {
            'transformers': True,  # Always available
            'vllm': False,
            'lmdeploy': False
        }
        
        try:
            import vllm
            compatibility['vllm'] = True
        except ImportError:
            pass
            
        try:
            import lmdeploy
            compatibility['lmdeploy'] = True
        except ImportError:
            pass
            
        return compatibility


class ImageProcessor:
    """Utility class for image processing."""
    
    @staticmethod
    def load_and_validate_images(image_paths: List[str]) -> Tuple[List[Image.Image], List[str]]:
        """
        Load and validate images from paths.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Tuple of (loaded images, error messages)
        """
        images = []
        errors = []
        
        for img_path in image_paths:
            try:
                if not os.path.exists(img_path):
                    errors.append(f"Image not found: {img_path}")
                    continue
                    
                img = Image.open(img_path)
                
                # Convert RGBA to RGB if necessary
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                    
                images.append(img)
                
            except Exception as e:
                errors.append(f"Error loading image {img_path}: {str(e)}")
                
        return images, errors
    
    @staticmethod
    def prepare_image_messages(images: List[Image.Image], max_pixels: int = 480 * 480) -> List[Dict]:
        """
        Prepare image messages for model input.
        
        Args:
            images: List of PIL Images
            max_pixels: Maximum pixels per image
            
        Returns:
            List of image message dictionaries
        """
        image_messages = []
        for img in images:
            image_messages.append({
                "type": "image",
                "image": img,
                "max_pixels": max_pixels
            })
        return image_messages


class ResponseProcessor:
    """Utility class for processing model responses."""
    
    @staticmethod
    def clean_response(response: str) -> str:
        """
        Clean and format model response.
        
        Args:
            response: Raw model response
            
        Returns:
            Cleaned response
        """
        if not response:
            return "No response generated"
            
        # Remove excessive whitespace
        response = re.sub(r'\s+', ' ', response.strip())
        
        # Basic validation - allow even single character answers like "A", "B" etc.
        if len(response) < 1:
            return "Error: Generated response too short"
            
        return response
    
    @staticmethod
    def extract_structured_response(response: str) -> Dict[str, str]:
        """
        Extract structured information from response.
        
        Args:
            response: Model response text
            
        Returns:
            Dictionary with extracted components
        """
        result = {'raw_response': response}
        
        # Extract CogMap section
        cogmap_match = re.search(r'<CogMap>\s*(.*?)\s*<', response, re.DOTALL)
        if cogmap_match:
            result['grounded_cogmap'] = cogmap_match.group(1).strip()
        
        # Extract Reasoning section
        reasoning_match = re.search(r'<Reasoning>\s*(.*?)\s*<', response, re.DOTALL)
        if reasoning_match:
            result['reasoning'] = reasoning_match.group(1).strip()
        
        # Extract Answer section
        answer_match = re.search(r'<Answer>\s*(.*?)$', response, re.DOTALL)
        if answer_match:
            result['answer'] = answer_match.group(1).strip()
        
        return result
    
    @staticmethod
    def validate_response(response: str, min_length: int = 10) -> Tuple[bool, str]:
        """
        Validate model response.
        
        Args:
            response: Model response
            min_length: Minimum response length
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not response:
            return False, "Empty response"
            
        if len(response.strip()) < min_length:
            return False, f"Response too short (< {min_length} characters)"
            
        return True, ""


class ConfigManager:
    """Utility class for managing configuration."""
    
    @staticmethod
    def get_default_config(model_type: str) -> Dict[str, Any]:
        """
        Get default configuration for a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Default configuration dictionary
        """
        configs = {
            'qwen2.5vl': {
                'max_pixels': 480 * 480,
                'max_new_tokens': 4096,
                'torch_dtype': 'float16',
                'device_map': 'auto',  # Will be optimized for full GPU usage
                'generation_config': {
                    'do_sample': False,    # Forces greedy decoding (deterministic)
                    'temperature': 0.0     # Backup for safety (ignored when do_sample=False)
                }
            },
            'gpt4v': {
                'max_tokens': 4096,
                'temperature': 0.0     # Forces deterministic inference for API models
            }
        }
        
        return configs.get(model_type, {})
    
    @staticmethod
    def load_config_from_file(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config from {config_path}: {e}")
            return {}
    
    @staticmethod
    def save_config_to_file(config: Dict[str, Any], config_path: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            config: Configuration dictionary
            config_path: Path to save config
        """
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Error saving config to {config_path}: {e}") 