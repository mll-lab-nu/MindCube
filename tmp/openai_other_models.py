from openai import OpenAI
import os
import base64
import json
import os
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from functools import partial
import logging
from typing import List, Dict, Any, Optional, Set

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('openai_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

updated_root = '/projects/b1222/userdata/qineng/01_projects/07_MindCube_new/data/'
file_path = "/projects/b1222/userdata/qineng/01_projects/07_MindCube_new/data/prompts/general/MindCube_tinybench_raw_qa.jsonl"
output_folder = "/projects/b1222/userdata/qineng/01_projects/07_MindCube_new/data/results/o3"
output_path = os.path.join(output_folder, "mindcube_o3_raw_qa_responses.jsonl")
model_id = "o3-2025-04-16"
NUM_PROCESSES = 64

def replace_old_root(image_paths):
    updated_path = [updated_root + image_path for image_path in image_paths]
    return updated_path

def open_image_with_exif(full_path):
    img = Image.open(full_path)
    try:
        # Use more standard PIL approach with getexif()
        exif = img.getexif() if hasattr(img, 'getexif') else None
        if exif is not None:
            orientation = exif.get(274, 1)  # 274 is the orientation tag
            if orientation == 3:  # 180 degree rotation
                img = img.rotate(180, expand=True)
            elif orientation == 6:  # Rotate 270 degrees
                img = img.rotate(270, expand=True)
            elif orientation == 8:  # Rotate 90 degrees
                img = img.rotate(90, expand=True)
    except Exception as e:
        print(f"Warning: Could not process EXIF data for {full_path}: {e}")
    return img

def find_json_files(input_root):
    json_files = []
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.endswith(".jsonl"):
                json_files.append(os.path.join(root, file))
    return json_files

def load_images(image_paths):
    images = []
    for path in image_paths:
        if os.path.exists(path):
            image = open_image_with_exif(path)

            ###RGBA
            if image.mode == 'RGBA':
                image = image.convert("RGB")
            images.append(image)
        else:
            print(f"Warning: Image path {path} does not exist.")
    return images

# Function to encode the image
def encode_image(image_path):
  try:
    # Open image with proper orientation handling
    img = open_image_with_exif(image_path)
    # Convert to RGB if needed
    if img.mode == 'RGBA':
      img = img.convert('RGB')
    # Save to a bytes buffer and encode
    import io
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')
  except Exception as e:
    print(f"Error encoding image {image_path}: {e}")
    # Fallback to direct file reading if there's an error
    with open(image_path, "rb") as image_file:
      return base64.b64encode(image_file.read()).decode('utf-8')

def test_image_path_readable(image_paths):
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"Warning: Image path {image_path} does not exist.")
            return False
        # try to open the image
        try:
            Image.open(image_path)
        except Exception as e:
            print(f"Error in opening image {image_path}: {e}")
            return False
    return True

def get_openai_response(model, prompt, image_paths):
    try:
        # Create a new client for each process to avoid sharing issues
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        base64_images = [encode_image(image) for image in image_paths]
        user_content = [{"type": "text", "text": prompt}]
        base64_images = [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail": "high",
                                    },
                                }
                                for base64_image in base64_images
                            ]
        user_content.extend(base64_images)
        messages = [{"role": "user", "content": user_content}]
        completion = client.chat.completions.create(
            model=model,
            messages=messages
        )
        model_answer = completion.choices[0].message.content
        return model_answer
    except Exception as e:
        logger.error(f"Error in get_openai_response: {e}")
        return None

def check_existing_results(output_path: str) -> Set[str]:
    """Check existing results and return set of completed IDs."""
    completed_ids = set()
    
    if not os.path.exists(output_path):
        logger.info(f"Output file {output_path} does not exist. Starting fresh.")
        return completed_ids
    
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    q_id = data.get("id", "")
                    if q_id:
                        completed_ids.add(q_id)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line {line_num}: {e}")
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
        
        logger.info(f"Found {len(completed_ids)} completed results in existing file")
        logger.info(f"Output file is readable: {output_path}")
        
    except Exception as e:
        logger.error(f"Error reading existing results: {e}")
        # If file exists but can't be read, backup and start fresh
        backup_path = output_path + f".backup_{int(time.time())}"
        try:
            os.rename(output_path, backup_path)
            logger.info(f"Backed up unreadable file to {backup_path}")
        except Exception:
            logger.error(f"Could not backup unreadable file")
    
    return completed_ids

def process_single_item(item_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a single data item. This function will be run in parallel."""
    try:
        q_id = item_data.get("id", "")
        question = item_data.get("input_prompt")
        image_paths = item_data.get("images", [])
        image_paths = replace_old_root(image_paths)
        
        # Validate image paths
        if not test_image_path_readable(image_paths):
            logger.error(f"Image paths not readable for {q_id}")
            return None
        
        # Get OpenAI response
        answer = get_openai_response(model_id, question, image_paths)
        
        if answer is None:
            logger.error(f"Failed to get response for {q_id}")
            return None
        
        # Prepare result
        result_data = item_data.copy()
        result_data["answer"] = answer
        result_data["processed_at"] = time.time()
        
        logger.info(f"Successfully processed {q_id}")
        return result_data
        
    except Exception as e:
        logger.error(f"Error processing item {item_data.get('id', 'unknown')}: {e}")
        return None

def write_result_safely(result: Dict[str, Any], output_path: str):
    """Safely write a single result to the output file."""
    try:
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()  # Ensure data is written immediately
    except Exception as e:
        logger.error(f"Error writing result for {result.get('id', 'unknown')}: {e}")

def main():
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Check existing results
    completed_ids = check_existing_results(output_path)
    
    # Load all testing data
    logger.info(f"Loading data from {file_path}")
    testing_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for idx, line in enumerate(tqdm(lines, desc="Loading data")):
                try:
                    data = json.loads(line.strip())
                    testing_data.append(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line {idx+1}: {e}")
    except Exception as e:
        logger.error(f"Error loading input file: {e}")
        return
    
    # Filter out already completed items
    pending_data = [item for item in testing_data if item.get("id", "") not in completed_ids]
    
    logger.info(f"Total items: {len(testing_data)}")
    logger.info(f"Already completed: {len(completed_ids)}")
    logger.info(f"Pending items: {len(pending_data)}")
    
    if not pending_data:
        logger.info("All items already processed. Exiting.")
        return
    
    # Process items with multiprocessing
    logger.info(f"Starting processing with {NUM_PROCESSES} processes")
    
    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        # Submit all pending tasks
        future_to_item = {executor.submit(process_single_item, item): item for item in pending_data}
        
        completed_count = 0
        failed_count = 0
        
        # Process completed tasks as they finish
        for future in tqdm(as_completed(future_to_item), total=len(pending_data), desc="Processing"):
            item = future_to_item[future]
            try:
                result = future.result()
                if result is not None:
                    # Write result immediately when it's ready
                    write_result_safely(result, output_path)
                    completed_count += 1
                else:
                    failed_count += 1
                    logger.warning(f"Failed to process item {item.get('id', 'unknown')}")
            except Exception as e:
                failed_count += 1
                logger.error(f"Exception processing item {item.get('id', 'unknown')}: {e}")
    
    logger.info(f"Processing completed. Success: {completed_count}, Failed: {failed_count}")
    logger.info(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()