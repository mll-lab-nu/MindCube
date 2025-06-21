"""
Base inference engine interface for MindCube project.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
import os
from tqdm import tqdm


class BaseInferenceEngine(ABC):
    """Base class for all inference engines."""
    
    def __init__(self, model_path: str, model_type: str = "qwen2.5vl", **kwargs):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the model
            model_type: Type of model (qwen2.5vl, gpt4v, etc.)
            **kwargs: Additional configuration
        """
        self.model_path = model_path
        self.model_type = model_type
        self.config = kwargs
        self.model = None
        self.processor = None
        
    @abstractmethod
    def load_model(self) -> None:
        """Load the model and processor."""
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def generate_response(self, processed_input: Any, **kwargs) -> str:
        """
        Generate response from processed input.
        
        Args:
            processed_input: Output from process_input
            **kwargs: Generation parameters
            
        Returns:
            Generated response text
        """
        pass
    
    def infer(self, prompt: str, image_paths: List[str], **kwargs) -> str:
        """
        Complete inference pipeline.
        
        Args:
            prompt: Text prompt
            image_paths: List of image file paths
            **kwargs: Additional parameters
            
        Returns:
            Generated response
        """
        try:
            # Load model if not already loaded (check both model types)
            if (getattr(self, 'model', None) is None) and \
               (getattr(self, 'vllm_engine', None) is None):
                self.load_model()
            
            # Process input
            processed_input = self.process_input(prompt, image_paths, **kwargs)
            
            # Generate response
            response = self.generate_response(processed_input, **kwargs)
            
            return response
            
        except Exception as e:
            return f"Error during inference: {str(e)}"
    
    def batch_infer(self, data_file: str, output_file: str, 
                   image_root: str = "./data/", batch_size: int = 1, **kwargs) -> None:
        """
        Batch inference from a data file with configurable batch size.
        
        Args:
            data_file: Path to input JSONL file
            output_file: Path to output JSONL file
            image_root: Root directory for images
            batch_size: Number of samples to process in each batch
            **kwargs: Additional parameters
        """
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Load all data first to show progress
        print(f"Loading data from {data_file}...")
        all_data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_data.append(json.loads(line.strip()))
        
        print(f"Processing {len(all_data)} samples with batch size {batch_size}...")
        
        # Clear output file
        with open(output_file, 'w', encoding='utf-8') as f:
            pass
        
        # Process data in batches
        successful_count = 0
        
        # Create batches with global progress bar
        for batch_start in tqdm(range(0, len(all_data), batch_size), desc="Global Batch Progress"):
            batch_end = min(batch_start + batch_size, len(all_data))
            batch_data = all_data[batch_start:batch_end]
            
            try:
                # Process batch
                batch_results = self.process_batch(batch_data, image_root, **kwargs)
                
                # Write results immediately
                with open(output_file, 'a', encoding='utf-8') as f:
                    for result in batch_results:
                        if result is not None:
                            f.write(json.dumps(result, ensure_ascii=False) + '\n')
                            successful_count += 1
                
                # Progress update
                print(f"Processed batch {batch_start//batch_size + 1}, samples {batch_start+1}-{batch_end}, successful: {successful_count}")
                
            except Exception as e:
                print(f"Error processing batch {batch_start//batch_size + 1}: {e}")
                # Still try to process individual samples in the batch
                for idx, data in enumerate(batch_data):
                    try:
                        result = self.process_single_sample(data, image_root, **kwargs)
                        if result is not None:
                            with open(output_file, 'a', encoding='utf-8') as f:
                                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                            successful_count += 1
                    except Exception as single_e:
                        print(f"Error processing sample {batch_start + idx}: {single_e}")
                        continue
        
        print(f"Batch inference completed! Processed {successful_count}/{len(all_data)} samples successfully.")
        print(f"Results saved to: {output_file}")

    def process_batch(self, batch_data: List[Dict], image_root: str, **kwargs) -> List[Optional[Dict]]:
        """
        Process a batch of samples. Can be overridden by subclasses for optimized batch processing.
        
        Args:
            batch_data: List of data samples
            image_root: Root directory for images
            **kwargs: Additional parameters
            
        Returns:
            List of results (None for failed samples)
        """
        # Default implementation: process each sample individually
        # Subclasses can override this for true batch processing
        results = []
        for data in batch_data:
            result = self.process_single_sample(data, image_root, **kwargs)
            results.append(result)
        return results
    
    def process_single_sample(self, data: Dict, image_root: str, **kwargs) -> Optional[Dict]:
        """
        Process a single sample.
        
        Args:
            data: Single data sample
            image_root: Root directory for images
            **kwargs: Additional parameters
            
        Returns:
            Result dictionary or None if failed
        """
        try:
            # Extract required fields
            prompt = data.get('input_prompt', '')
            image_paths = data.get('images', [])
            
            if not prompt:
                print(f"Warning: No input_prompt found for sample")
                return None
            
            # Update image paths with root directory - FORCE all paths to use image_root
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
            
            # Validate image paths
            valid_images = []
            for img_path in image_paths:
                if os.path.exists(img_path):
                    valid_images.append(img_path)
                else:
                    print(f"Warning: Image not found: {img_path}")
            
            # Generate response
            response = self.infer(prompt, valid_images, **kwargs)
            
            # Prepare output in the same format as original script
            result = data.copy()
            result['answer'] = response  # Use 'answer' field like original
            result['model_type'] = self.model_type
            
            return result
            
        except Exception as e:
            print(f"Error processing sample: {e}")
            return None
    
    def validate_inputs(self, prompt: str, image_paths: List[str]) -> bool:
        """
        Validate input data.
        
        Args:
            prompt: Text prompt
            image_paths: List of image paths
            
        Returns:
            True if inputs are valid
        """
        if not prompt or not prompt.strip():
            print("Warning: Empty prompt")
            return False
            
        for img_path in image_paths:
            if not os.path.exists(img_path):
                print(f"Warning: Image path {img_path} does not exist")
                return False
                
        return True 