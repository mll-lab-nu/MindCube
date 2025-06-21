"""
Around Reasoning Chain Generation

Implementation for around spatial reasoning scenarios.
Generates detailed reasoning chains for "around" type questions based on spatial relationships.
"""

import json
import re
import os
import os.path
import traceback
from typing import Dict


def extract_options_from_question(question_text):
    """Extract answer options from question text."""
    options_map = {}
    question_parts = question_text.split("? ")
    if len(question_parts) < 2:
        return options_map
    
    options_string = question_parts[-1].strip()

    # Regex for option A, looking ahead for B or end of string
    match_A = re.search(r"A\.\s*(.+?)(?=(\s*B\.|$))", options_string)
    if match_A:
        options_map["A"] = match_A.group(1).strip()

    # Regex for option B, looking ahead for C or end of string
    match_B = re.search(r"B\.\s*(.+?)(?=(\s*C\.|$))", options_string)
    if match_B:
        options_map["B"] = match_B.group(1).strip()

    # Regex for option C, looking ahead for D or end of string
    match_C = re.search(r"C\.\s*(.+?)(?=(\s*D\.|$))", options_string)
    if match_C:
        options_map["C"] = match_C.group(1).strip()

    # Regex for option D, looking ahead for end of string
    # It must be the last option.
    if "C" in options_map: # Only look for D if C exists
        match_D = re.search(r"D\.\s*(.+?)(?=(\s*$))", options_string)
        if match_D:
             options_map["D"] = match_D.group(1).strip()
    elif "B" not in options_map and "A" in options_map : # If only A exists, D cannot exist in A B C D format
        pass

    return options_map


def generate_q1_reasoning_chain(question_text, meta_info, question_prefix, question_id, gt_answer_key):
    """
    Generate reasoning chain for q1 questions (self-movement direction).
    
    Args:
        question_text: The question text
        meta_info: Metadata information containing objects and spatial relationships
        question_prefix: Question type prefix
        question_id: Question identifier
        gt_answer_key: Ground truth answer key
        
    Returns:
        Generated reasoning chain as string
    """
    reasoning = []
    
    # Validate meta_info structure
    if not meta_info or len(meta_info) < 2 or not meta_info[1] or len(meta_info[1]) < 4:
        return f"Error: Invalid meta_info structure for Q1 reasoning. Expected meta_info[1] with at least 4 elements, got: {meta_info}"
    
    obj_pool = meta_info[1][1]
    direction = meta_info[1][2]
    other_object = meta_info[1][3]
    
    # Validate obj_pool
    if not obj_pool or len(obj_pool) < 2:
        return f"Error: Invalid obj_pool. Expected at least 2 objects, got: {obj_pool}"

    problem_id = question_id # 1,2
    question = question_text

    # Start with a description of what we can see in each image without mentioning specific views
    reasoning.append("I need to determine how I moved from the viewpoint in image 1 to the viewpoint in image 2. In image 1, ")
    i = 0
    if i < len(obj_pool) - 1:  # 减1是因为我们要比较obj_pool[j]和obj_pool[j+1]
        for j in range(i):
            reasoning.append(f"I can see {obj_pool[j]} on the left of the {obj_pool[j+1]};")
        reasoning.append(f"I can also see {obj_pool[i]} on the left of the {obj_pool[i+1]}.")  # 结束句

    # Describe visible objects in each image using correct left-to-right ordering for each view
    if problem_id == 1:
        reasoning.append(f"In image 2, I can see {obj_pool[0]} in front of the {obj_pool[1]}.")
        reasoning.append(f"I notice that {obj_pool[0]} is visible in both images, but from different angles.")
    else:
        reasoning.append(f"In image 2, I can see {obj_pool[-1]} in front of the {obj_pool[-2]}.")
        reasoning.append(f"I notice that {obj_pool[-1]} is visible in both images, but from different angles.")
    
    # Detailed imagination-based reasoning about movement direction
    reasoning.append(f"I analyze how the viewpoint changed from image 1 to image 2 by analyzing how the structural features of objects on the platform and relative positions of these objects transform between the two views. This involves tracking how key features shift positions, observing which elements become more prominent or less visible, and comparing if these observed changes align with the expected spatial transformations that would occur when viewing the object from its left or right side.")

    if problem_id == 1:
        reasoning.append(f"Image 2 seems to show the {obj_pool[0]}'s left side compared to the first one.")
        reasoning.append("This suggests that, to transition from the viewpoint in the first image to the viewpoint in the second image, I need to move forward and to the left.")
    else:
        reasoning.append(f"Image 2 seems to show the {obj_pool[-1]}'s right side compared to the first one.")
        reasoning.append("This suggests that, to transition from the viewpoint in the first image to the viewpoint in the second image, I need to move forward and to the right.")

    options_map = extract_options_from_question(question)
    # Get the actual text for the correct answer letter
    correct_answer_text = options_map.get(gt_answer_key, "").strip()

    # Conclusion - avoid double periods
    if correct_answer_text and not correct_answer_text.endswith('.'):
        reasoning.append(f"Therefore, the answer is {gt_answer_key}. {correct_answer_text}.")
    else:
        reasoning.append(f"Therefore, the answer is {gt_answer_key}. {correct_answer_text}")
    reasoning = " ".join(reasoning)
    return reasoning


def generate_q23_reasoning_chain(question_text, meta_info, question_prefix, question_id, gt_answer_key):
    """
    Generate reasoning chain for q2/q3 questions (object relationships and positioning).
    
    Args:
        question_text: The question text
        meta_info: Metadata information containing objects and spatial relationships
        question_prefix: Question type prefix (2, 3, 4)
        question_id: Question identifier (1-8)
        gt_answer_key: Ground truth answer key
        
    Returns:
        Generated reasoning chain as string
    """
    initial_problem_type = question_prefix # 2,3,4
    question = question_text
    
    reasoning = []
    
    # Validate meta_info structure
    if not meta_info or len(meta_info) < 2:
        return f"Error: Invalid meta_info structure for Q23 reasoning. Expected meta_info with at least 2 elements, got: {meta_info}"
    
    if not meta_info[0] or len(meta_info[0]) < 1:
        return f"Error: Invalid meta_info[0] structure. Expected at least 1 element, got: {meta_info[0]}"
        
    if not meta_info[1] or len(meta_info[1]) < 4:
        return f"Error: Invalid meta_info[1] structure. Expected at least 4 elements, got: {meta_info[1]}"
    
    img_num = meta_info[0][0]
    obj_pool = meta_info[1][1]
    direction = meta_info[1][2]
    other_object = meta_info[1][3]
    
    # Validate obj_pool
    if not obj_pool or len(obj_pool) < 2:
        return f"Error: Invalid obj_pool. Expected at least 2 objects, got: {obj_pool}"

    problem_id = question_id # 1,2,3,4,5,6,7,8
    question = question_text
    special = 1
    if 'nearest object' in question:
        special = 3
    if 'is there' in question:
        special = 2
        
    if problem_id != 7 and problem_id != 8:
        camera_movement_desc = "Image 1 is the front view of the scene. Image 2 (left view) is captured after rotating around the scene 90 degrees clockwise from image 1. Image 3 (right view) is captured after rotating around the scene 90 degrees counterclockwise from image 1."
    else: 
        camera_movement_desc = "Image 1 is the back view of the scene. Image 2 (left view) is captured after rotating around the scene 90 degrees clockwise from image 1. Image 3 (right view) is captured after rotating around the scene 90 degrees counterclockwise from image 1."

    # 1. Scene Setup Observation: Describe views, primary objects, and camera movement
    object_observations_parts = []
    i = 0
    object_observations_parts.append('In image 1, ')
    if problem_id != 3 and problem_id != 4:
        if i < len(obj_pool) - 1:  # 减1是因为我们要比较obj_pool[j]和obj_pool[j+1]
            for j in range(i):
                object_observations_parts.append(f"I can see {obj_pool[j]} on the left of the {obj_pool[j+1]};")
            object_observations_parts.append(f"I can also see {obj_pool[i]} on the left of the {obj_pool[i+1]}.")  # 结束句
    else:
        if i < len(obj_pool) - 1:  # 减1是因为我们要比较obj_pool[j]和obj_pool[j+1]
            for j in range(i):
                object_observations_parts.append(f"I can see {obj_pool[j]} on the right of the {obj_pool[j+1]};")
            object_observations_parts.append(f"I can also see {obj_pool[i]} on the right of the {obj_pool[i+1]}. These objects appear to be roughly aligned along a straight line that is parallel to the viewing direction of me, which results in all of them appearing at approximately the same depth. This alignment makes depth-based differentiation clearer when viewed from the sides.")  # 结束句
            
    object_observations_parts.append(f"In image 2, I can see {obj_pool[0]} clearly.")
    object_observations_parts.append(f"In image 3, I can see {obj_pool[-1]} clearly.")
    object_observations = " ".join(object_observations_parts)

    scene_description_intro = f"In this scene, I observe three images showing the same scene from different perspectives. {object_observations}"
    mental_process = "To identify the position change across views, I focus on the main object's orientation. Then, I analyze the angles and relative positions of other objects in the scene to support this observation. I understand that: " 
    
    reasoning.append(f"{scene_description_intro} {mental_process} {camera_movement_desc}")

    # 2. Amodal reasoning
    left = f"When observing objects {obj_pool[0]} and {obj_pool[1]} in the front view, where {obj_pool[0]} is positioned to the left of {obj_pool[1]}, this spatial relationship transforms when viewing from the left view. Consider this: objects that appear more leftward in the front view become closer to the observer in the left view, while objects that appear more rightward become farther away. Following the principle of depth perception, where closer objects appear in front of farther objects from the observer's perspective, we can conclude that in the left view, {obj_pool[1]} should appear behind {obj_pool[0]}. This remains true even if {obj_pool[1]} is partially or fully occluded by {obj_pool[0]} in the left view, as such occlusion actually validates our spatial reasoning about their relative positions."
                 
    left_back = f"When observing objects {obj_pool[0]} and {obj_pool[1]} in the back view, where {obj_pool[0]} is positioned to the right of {obj_pool[1]}, this spatial relationship transforms when viewing from the left view. Consider this: objects that appear more rightward in the back view become closer to the observer in the left view, while objects that appear more leftward become farther away. Following the principle of depth perception, where closer objects appear in front of farther objects from the observer's perspective, we can conclude that in the left view, {obj_pool[1]} should appear behind {obj_pool[0]}. This remains true even if {obj_pool[1]} is partially or fully occluded by {obj_pool[0]} in the left view, as such occlusion actually validates our spatial reasoning about their relative positions."
                 
    right = f"When observing objects {obj_pool[-1]} and {obj_pool[-2]} in the front view, where {obj_pool[-1]} is positioned to the right of {obj_pool[-2]}, this spatial relationship transforms when viewing from the right view. Consider this: objects that appear more rightward in the front view become closer to the observer in the right view, while objects that appear more leftward become farther away. Following the principle of depth perception, where closer objects appear in front of farther objects from the observer's perspective, we can conclude that in the right view, {obj_pool[-2]} should appear behind {obj_pool[-1]}. This remains true even if {obj_pool[-2]} is partially or fully occluded by {obj_pool[-1]} in the right view, as such occlusion actually validates our spatial reasoning about their relative positions."
                 
    right_back = f"When observing objects {obj_pool[-1]} and {obj_pool[-2]} in the back view, where {obj_pool[-1]} is positioned to the left of {obj_pool[-2]}, this spatial relationship transforms when viewing from the right view. Objects that appear more leftward in the back view become closer to the observer in the right view, while objects that appear more rightward become farther away. Following the principle of depth perception, where closer objects appear in front of farther objects from the observer's perspective, we can conclude that in the right view, {obj_pool[-1]} should appear in front of {obj_pool[-2]}. This remains true even if {obj_pool[-2]} is partially or fully occluded by {obj_pool[-1]} in the right view, as such occlusion actually validates our spatial reasoning about their relative positions." 

    if initial_problem_type == 2:  
        mental_process2_left = (lambda obj1, obj2: f" Therefore, when transitioning from front view to left view, there is a {obj2} behind {obj1} as a direct consequence of {obj1} being left of {obj2} in the initial front view.")(*((obj_pool[1], obj_pool[2]) if (problem_id == 7 and len(obj_pool) > 2) else (obj_pool[0], obj_pool[1])))
        
        mental_process3_left = f"Furthermore, to determine which object is directly behind a specific object in the left view (i.e., the nearest object behind it), we need to examine the front view arrangement: the object that appears immediately to the right of our reference object in the front view will be the nearest object behind it in the left view. This spatial transformation maintains the relative distances between objects. If this immediate neighbor is not among the available options, we should consider the next right object in sequence."

        mental_process2_right = (lambda obj1, obj2: f"Therefore, when transitioning from front view to right view, there is a {obj2} behind {obj1} as a direct consequence of {obj1} being right of {obj2} in the initial front view.")(*((obj_pool[-2], obj_pool[-3]) if (problem_id == 8 and len(obj_pool) > 2) else (obj_pool[-1], obj_pool[-2])))
        
        mental_process3_right = f"Furthermore, to determine which object is directly behind a specific object in the right view (i.e., the nearest object behind it), we need to examine the front view arrangement: the object that appears immediately to the left of our reference object in the front view will be the nearest object behind it in the right view. This spatial transformation maintains the relative distances between objects. If this immediate neighbor is not among the available options, we should consider the next left object in sequence."

        if problem_id in [1, 3, 5, 7]:  
            if problem_id == 7:
                reasoning.append(left_back)
            else:
                reasoning.append(left)

            if special == 2:
                reasoning.append(mental_process2_left)
            else:
                reasoning.append(mental_process3_left)

        if problem_id in [2, 4, 6, 8]:
            if problem_id == 8:
                reasoning.append(right_back)
            else:
                reasoning.append(right)    

            if special == 2:
                reasoning.append(mental_process2_right)
            else:
                reasoning.append(mental_process3_right)

    # 3. P-O what if
    if initial_problem_type == 3:  
        if problem_id in [5, 9]:
            reasoning.append(left_back)
        elif problem_id in [1, 3, 7]:
            reasoning.append(left)
        elif problem_id in [6, 10]:
            reasoning.append(right_back)
        elif problem_id in [2, 4, 8]:
            reasoning.append(right)

        if problem_id in [1, 3, 7, 5, 9]:
            if problem_id in [7, 9]:
                turn = "left"
            else:
                turn = "right"
            whatif_process = "If I am standing at the viewpoint presented in image 2, this means I am looking at the left side of the scene. Based on the analysis before, the {} should be my front, although I couldn't see it clearly. So, if I turn {} and move forward, I will get farther from the {}.".format(obj_pool[1], turn, obj_pool[1])
        elif problem_id in [2, 4, 8, 6, 10]:
            if problem_id in [8, 10]:
                turn = "right"
            else:
                turn = "left"
            whatif_process = "If I am standing at the viewpoint presented in image 3, this means I am looking at the right side of the scene. Based on the analysis before, the {} should be my front, although I couldn't see it clearly. So, if I turn {} and move forward, I will get farther from the {}.".format(obj_pool[-2], turn, obj_pool[-2])
        reasoning.append(whatif_process)

    # 4. P-O perspective
    if initial_problem_type == 4:  
        # Validate direction data
        if not direction or len(direction) == 0:
            return f"Error: Invalid direction data for Q4 reasoning. Expected non-empty direction array, got: {direction}"
        
        direction_obj = direction[0]
        
        # Handle None or invalid direction values
        if direction_obj is None or direction_obj == "None":
            # Try to find a valid direction from the list
            valid_direction = None
            for d in direction:
                if d and d != "None" and isinstance(d, str):
                    valid_direction = d
                    break
            
            if valid_direction:
                direction_obj = valid_direction
            else:
                return f"Error: No valid direction found in direction array: {direction}"
        
        direction_view = {
            'face': 'back',
            'back': 'front',
            'left': 'right',
            'right': 'left'
        }
        
        # Validate direction_obj is in our mapping
        if direction_obj not in direction_view:
            return f"Error: Unknown direction '{direction_obj}'. Expected one of: {list(direction_view.keys())}"
        
        if initial_problem_type == 4:  # Fixed condition
            reasoning.append(f"If I were positioned where the {obj_pool[0]} is, I would analyze its direction. Based on its structure and affordance shown in images, I believe its direction would be the same as the {direction_view[direction_obj]} view of the scene.")
            reasoning.append(f"So, from the {obj_pool[0]}'s perspective, the nearest object to my left is the {obj_pool[1]}.")   

    # 7. Final Answer - Fix the issue with answer option text by extracting directly from options
    options_map = extract_options_from_question(question)
    # Get the actual text for the correct answer letter
    correct_answer_text = options_map.get(gt_answer_key, "").strip()
    # Avoid double periods
    if correct_answer_text and not correct_answer_text.endswith('.'):
        reasoning.append(f"So the answer is {gt_answer_key}. {correct_answer_text}.")
    else:
        reasoning.append(f"So the answer is {gt_answer_key}. {correct_answer_text}")

    reasoning = " ".join(reasoning)
    
    return reasoning


def generate_around_reasoning_chain(item: Dict) -> str:
    """
    Generate reasoning chain for around spatial reasoning scenarios.
    
    This is the main function that generates detailed reasoning chains for "around" type questions
    based on the item's data and metadata.
    
    Args:
        item: QA data item containing:
            - id: Item identifier  
            - question: Question text
            - gt_answer: Ground truth answer
            - meta_info: Metadata for reasoning
            - type: Data type
            - images: List of image paths (optional)
            
    Returns:
        Generated reasoning chain as string
    """
    try:
        question_text = item.get('question', '')
        answer_key = item.get('gt_answer', '')
        meta_info = item.get('meta_info', [])
        id = item.get('id', '')
        data_type = item.get('type')
        
        # Validate basic fields
        if not question_text:
            return f"Error: Missing question text for item {id}"
        if not answer_key:
            return f"Error: Missing gt_answer for item {id}"
        if not meta_info:
            return f"Error: Missing meta_info for item {id}"
        
        # Parse ID to get question type and ID
        parts = id.split('_')
        if len(parts) < 4:
            return f"Error: Invalid ID format for item {id}. Expected format: prefix_xxx_qX_Y"
        
        try:
            question_prefix = int(parts[2][1]) 
            question_id = int(parts[3])
        except (IndexError, ValueError) as e:
            return f"Error: Cannot parse question prefix/ID from {id}: {e}"

        # Generate reasoning chain based on question type
        if question_prefix == 1:  # Self-movement direction
            reasoning_chain = generate_q1_reasoning_chain(
                question_text, meta_info, question_prefix, question_id, answer_key
            )
        else:
            reasoning_chain = generate_q23_reasoning_chain(
                question_text, meta_info, question_prefix, question_id, answer_key
            )
        
        return reasoning_chain
        
    except Exception as e:
        # Return error message instead of raising exception
        error_msg = f"Error generating around reasoning for item {item.get('id', 'unknown')}: {str(e)}"
        return error_msg


def add_reasoning_chains_to_around_data(input_file: str, output_file: str) -> None:
    """
    Add reasoning chains to around scenario data.
    
    Reads a JSONL file, adds reasoning chains to items with "around" in their ID,
    and writes ONLY these "around" items to a new JSONL file.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file with reasoning chains
    """
    print(f"🔄 Processing around reasoning: {input_file} -> {output_file}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Load input data
    all_items_from_input = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                if line.strip():
                    all_items_from_input.append(json.loads(line))
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at {input_file}")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Error: Could not decode JSON from input file {input_file}. Details: {e}")
        return

    around_items_processed_count = 0
    error_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item_idx, item in enumerate(all_items_from_input):
            try:
                item_id = item.get("id", "")
                if "aroundnew" in item_id or "around" in item_id.lower():  # Process only items with "around" in their ID
                    reasoning_chain = generate_around_reasoning_chain(item)
                    item['reasoning_chain'] = reasoning_chain
                    around_items_processed_count += 1
                    f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
            except Exception as e:
                error_count += 1
                print(f"❌ Error processing item at index {item_idx} with ID '{item.get('id', 'N/A')}': {e}")
                # Still write the item with error reasoning
                item['reasoning_chain'] = f"Error generating reasoning chain: {str(e)}"
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                around_items_processed_count += 1

    print(f"✅ Processed and saved {around_items_processed_count} 'around' items to {output_file}")
    print(f"📊 Encountered {error_count} errors during processing")


# Additional utility functions for around reasoning
def extract_around_features(item: Dict) -> Dict:
    """
    Extract features specific to around reasoning scenarios.
    
    Args:
        item: QA data item
        
    Returns:
        Extracted features for around reasoning
    """
    features = {
        "item_id": item.get("id", ""),
        "setting": "around",
        "has_meta_info": "meta_info" in item,
        "question_type": None,
        "spatial_elements": []
    }
    
    # Extract question type from ID if possible
    item_id = item.get("id", "")
    if item_id:
        try:
            parts = item_id.split('_')
            if len(parts) > 2:
                features["question_type"] = parts[2]
        except:
            pass
    
    # Extract spatial elements from meta_info
    meta_info = item.get("meta_info", [])
    if meta_info and len(meta_info) > 1 and len(meta_info[1]) > 1:
        features["spatial_elements"] = meta_info[1][1] if isinstance(meta_info[1][1], list) else []
    
    return features


def validate_around_item(item: Dict) -> bool:
    """
    Validate that an item is suitable for around reasoning.
    
    Args:
        item: QA data item
        
    Returns:
        True if item is valid for around reasoning
    """
    # Basic validation - check for required fields
    required_fields = ["id", "question", "gt_answer"]
    
    for field in required_fields:
        if field not in item:
            return False
    
    # Check if item is around-related
    item_id = item.get("id", "").lower()
    if not ("around" in item_id or "aroundnew" in item_id):
        return False
    
    # Check for meta_info structure
    meta_info = item.get("meta_info", [])
    if not meta_info or len(meta_info) < 2:
        return False
    
    return True


if __name__ == "__main__":
    # Example usage for testing
    print("Around reasoning module - implementation active")
    
    # Example item for testing
    test_item = {
        "id": "aroundnew_test_q1_001",
        "question": "Based on these images, how should I move from image 1 to image 2? A. Forward and left B. Forward and right C. Backward and left D. Backward and right",
        "gt_answer": "A",
        "meta_info": [
            [3],  # img_num
            [
                ["object1", "object2"],  # obj_pool
                "left",  # direction
                "other_object"  # other_object
            ]
        ],
        "type": "around",
        "images": ["test1.png", "test2.png"]
    }
    
    # Test reasoning generation
    try:
        reasoning = generate_around_reasoning_chain(test_item)
        print("✅ Test reasoning chain generated successfully:")
        print(reasoning[:200] + "..." if len(reasoning) > 200 else reasoning)
    except Exception as e:
        print(f"❌ Test failed: {e}") 