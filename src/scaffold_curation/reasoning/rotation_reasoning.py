import json
import re
import os

# Helper function to extract A, B, C, D options from the question text
def extract_options_from_question(question_text):
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

# Helper function to determine initial object layout relative to a camera viewpoint
def determine_initial_object_layout(viewpoint_idx, meta_info, scene_type_from_item):
    layout = {"front": None, "right": None, "behind": None, "left": None}
    num_objects = len(meta_info)

    # Helper to safely access meta_info
    m = lambda i: meta_info[i] if i < num_objects else None

    if scene_type_from_item == 'three_view': # 3 objects m(0), m(1), m(2)
        if viewpoint_idx == 1: # View 1: Front=m(0), Right=m(1), Behind=m(2)
            layout = {"front": m(0), "right": m(1), "behind": m(2), "left": None}
        elif viewpoint_idx == 2: # View 2: Front=m(1), Right=m(2), Left=m(0)
            layout = {"front": m(1), "right": m(2), "behind": None, "left": m(0)}
        elif viewpoint_idx == 3: # View 3: Front=m(2), Behind=m(0), Left=m(1)
            layout = {"front": m(2), "right": None, "behind": m(0), "left": m(1)}
    elif scene_type_from_item == 'two_view_clockwise': # 2 objects m(0), m(1)
        if viewpoint_idx == 1: # View 1: Front=m(0), Right=m(1)
            layout = {"front": m(0), "right": m(1), "behind": None, "left": None}
        elif viewpoint_idx == 2: # View 2: Front=m(1), Left=m(0)
            layout = {"front": m(1), "right": None, "behind": None, "left": m(0)}
    elif scene_type_from_item == 'two_view_counterclockwise': # 2 objects m(0), m(1)
        if viewpoint_idx == 1: # View 1: Front=m(0), Left=m(1)
            layout = {"front": m(0), "right": None, "behind": None, "left": m(1)}
        elif viewpoint_idx == 2: # View 2: Front=m(1), Right=m(0)
            layout = {"front": m(1), "right": m(0), "behind": None, "left": None}
    elif scene_type_from_item == 'two_view_opposite': # 2 objects m(0), m(1)
        if viewpoint_idx == 1: # View 1: Front=m(0), Behind=m(1)
            layout = {"front": m(0), "right": None, "behind": m(1), "left": None}
        elif viewpoint_idx == 2: # View 2: Front=m(1), Behind=m(0)
            layout = {"front": m(1), "right": None, "behind": m(0), "left": None}
    elif scene_type_from_item == 'four_view': # 4 objects m(0), m(1), m(2), m(3)
        if viewpoint_idx == 1: # F=m(0), R=m(1), B=m(2), L=m(3)
            layout = {"front": m(0), "right": m(1), "behind": m(2), "left": m(3)}
        elif viewpoint_idx == 2: # F=m(1), R=m(2), B=m(3), L=m(0)
            layout = {"front": m(1), "right": m(2), "behind": m(3), "left": m(0)}
        elif viewpoint_idx == 3: # F=m(2), R=m(3), B=m(0), L=m(1)
            layout = {"front": m(2), "right": m(3), "behind": m(0), "left": m(1)}
        elif viewpoint_idx == 4: # F=m(3), R=m(0), B=m(1), L=m(2)
            layout = {"front": m(3), "right": m(0), "behind": m(1), "left": m(2)}
    return layout

# Helper function to apply a mental turn to the current layout
def apply_turn(current_layout, turn_description_str):
    new_layout = current_layout.copy()
    if not turn_description_str or turn_description_str == "none":
        return new_layout

    old_front = current_layout["front"]
    old_right = current_layout["right"]
    old_behind = current_layout["behind"]
    old_left = current_layout["left"]

    if "90 degrees to the left" in turn_description_str:
        new_layout["front"] = old_left
        new_layout["right"] = old_front
        new_layout["behind"] = old_right
        new_layout["left"] = old_behind
    elif "90 degrees to the right" in turn_description_str:
        new_layout["front"] = old_right
        new_layout["right"] = old_behind
        new_layout["behind"] = old_left
        new_layout["left"] = old_front
    elif "180 degrees around" in turn_description_str: # "around" implies a 180-degree turn
        new_layout["front"] = old_behind
        new_layout["right"] = old_left
        new_layout["behind"] = old_front
        new_layout["left"] = old_right
    return new_layout

def generate_rotation_reasoning_chain(item):
    question = item['question']
    gt_answer_key = item['gt_answer']
    meta_info = item['meta_info']
    scene_type = item['type'] # e.g., 'three_view', 'two_view_clockwise'
    
    reasoning = []

    # 1. Scene Setup Observation: Describe views, primary objects, and camera movement
    num_objects = len(meta_info)
    object_observations_parts = []
    if num_objects > 0: object_observations_parts.append(f"In image 1, I can see {meta_info[0]} as the main object in front of me.")
    if num_objects > 1: object_observations_parts.append(f"In image 2, I can see {meta_info[1]} as the main object in front of me.")
    if num_objects > 2: object_observations_parts.append(f"In image 3, I can see {meta_info[2]} as the main object in front of me.")
    if num_objects > 3: object_observations_parts.append(f"In image 4, I can see {meta_info[3]} as the main object in front of me.")
    object_observations = " ".join(object_observations_parts)

    scene_description_intro = ""
    camera_movement_desc = ""
    if scene_type == 'three_view':
        scene_description_intro = f"This scene is observed using three images. {object_observations}"
        camera_movement_desc = "Image 1 is the initial view. Image 2 is captured after a 90-degree clockwise rotation from image 1. Image 3 is captured after another 90-degree clockwise rotation from image 2."
    elif scene_type == 'two_view_clockwise':
        scene_description_intro = f"This scene is observed using two images. {object_observations}"
        camera_movement_desc = "Image 1 is the initial view. Image 2 is captured after a 90-degree clockwise rotation from image 1."
    elif scene_type == 'two_view_counterclockwise':
        scene_description_intro = f"This scene is observed using two images. {object_observations}"
        camera_movement_desc = "Image 1 is the initial view. Image 2 is captured after a 90-degree counter-clockwise rotation from image 1."
    elif scene_type == 'two_view_opposite':
        scene_description_intro = f"This scene is observed using two images. {object_observations}"
        camera_movement_desc = "Image 1 is the initial view. Image 2 is captured from the opposite direction (a 180-degree rotation) relative to image 1."
    elif scene_type == 'four_view':
        scene_description_intro = f"This scene is observed using four images. {object_observations}"
        camera_movement_desc = "Image 1 is the initial view. Image 2 is captured after a 90-degree clockwise rotation from image 1. Image 3 is after another 90-degree clockwise rotation (180 degrees from image 1). Image 4 is after a further 90-degree clockwise rotation (270 degrees from image 1)."
    reasoning.append(f"{scene_description_intro} {camera_movement_desc}")

    # 2. Question Interpretation: Parse viewpoint, turn, and target direction
    # Look for both "viewpoint presented in image X" and "facing the same direction as shown in image X"
    initial_viewpoint_idx_match = re.search(r"(?:viewpoint presented in|facing the same direction as shown in) image (\d+)", question)
    initial_viewpoint_idx = int(initial_viewpoint_idx_match.group(1)) if initial_viewpoint_idx_match else 1

    turn_description = "none" # Default to no turn
    # Look for various turn patterns: "and turn", "then I turn", "then turn", etc.
    turn_match = re.search(r"(?:and|then)(?: I)? turn (90 degrees to the left|90 degrees to the right|180 degrees around)", question, re.IGNORECASE)
    if turn_match:
        turn_description = turn_match.group(1).lower()

    # Look for various question patterns: "what is to my X", "what is to your X", etc.
    target_relative_direction_match = re.search(r"what is to (?:my|your) (right|left|behind|front)\??", question, re.IGNORECASE)
    if not target_relative_direction_match: # Alternative phrasing: "what is (direction) you?"
        target_relative_direction_match = re.search(r"what is (right|left|behind|front) (?:you|me)\??", question, re.IGNORECASE)
    
    target_relative_direction = target_relative_direction_match.group(1).lower() if target_relative_direction_match else "unknown_direction"

    # 3. Establishing Initial Orientation from the specified viewpoint
    base_layout = determine_initial_object_layout(initial_viewpoint_idx, meta_info, scene_type)
    
    base_layout_desc_parts = []
    if base_layout["front"]: base_layout_desc_parts.append(f"'{base_layout['front']}' is in front")
    if base_layout["right"]: base_layout_desc_parts.append(f"'{base_layout['right']}' is to the right")
    if base_layout["behind"]: base_layout_desc_parts.append(f"'{base_layout['behind']}' is behind")
    if base_layout["left"]: base_layout_desc_parts.append(f"'{base_layout['left']}' is to the left")
    reasoning.append(f"From the perspective of image {initial_viewpoint_idx}: {', '.join(base_layout_desc_parts)}.")

    # 4. Simulating Mental Turn (if any)
    final_layout = base_layout
    if turn_description != "none":
        final_layout = apply_turn(base_layout, turn_description)
        
        turned_layout_desc_parts = []
        if final_layout["front"]: turned_layout_desc_parts.append(f"'{final_layout['front']}' is now in front")
        if final_layout["right"]: turned_layout_desc_parts.append(f"'{final_layout['right']}' is now to my right")
        if final_layout["behind"]: turned_layout_desc_parts.append(f"'{final_layout['behind']}' is now behind")
        if final_layout["left"]: turned_layout_desc_parts.append(f"'{final_layout['left']}' is now to my left")
        reasoning.append(f"After turning {turn_description}: {', '.join(turned_layout_desc_parts)}.")

    # 5. Identifying the Object in the Target Direction
    object_in_target_direction = final_layout.get(target_relative_direction)

    # 6. Conclusion
    if object_in_target_direction:
        reasoning.append(f"The object located to my {target_relative_direction} is '{object_in_target_direction}'.")
        
        conclusion_stmt_parts = [f"Therefore, from the viewpoint of image {initial_viewpoint_idx}"]
        if turn_description != "none":
            conclusion_stmt_parts.append(f", after a mental turn of {turn_description},")
        conclusion_stmt_parts.append(f" the object to my {target_relative_direction} is '{object_in_target_direction}'.")
        reasoning.append("".join(conclusion_stmt_parts))
    else:
        reasoning.append(f"Based on the layout, no specific object from the list is identified to my {target_relative_direction}.")

    # 7. Final Answer - Fix the issue with answer option text by extracting directly from options
    options_map = extract_options_from_question(question)
    # Get the actual text for the correct answer letter
    correct_answer_text = options_map.get(gt_answer_key, "").strip()
    reasoning.append(f"The answer is {gt_answer_key}. {correct_answer_text}")
    
    return reasoning


def add_reasoning_chains_to_data(input_file, output_file):
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    all_items = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_items.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        print("Creating a dummy input file for demonstration if specific paths are not available.")
        # This dummy data is for local testing if the file is missing.
        all_items = [
            {"id": "rotation_group000_q1_1", "category": ["perpendicular", "PP", "rotation", "self"], "type": "three_view", "meta_info": ["chair and white board", "door", "two chairs"], "question": "These three images (image 1, 2, and 3) show the same scene from three different viewpoints. The image 2 was taken after turning the camera 90 degrees to the right (clockwise) from the position of image 1. For image 3, the camera was turned another 90 degrees right, so it's basically facing the opposite direction of image 1. Based on these three images: If you are standing at the viewpoint presented in image 1, what is to your right? A. Chair and white board B. Door C. Two chairs", "images": ["img1.png", "img2.png", "img3.png"], "gt_answer": "B"},
            {"id": "rotation_group002_qfirst_5", "category": ["perpendicular", "PP", "rotation", "self"], "type": "two_view_clockwise", "meta_info": ["blue bin", "black sofa"], "question": "These two images (image 1 and 2) show the same scene from two different viewpoints. Image 2 was taken after turning the camera 90 degrees to the right (clockwise) from the position of image 1. Based on these two images: If you are standing at the viewpoint presented in image 1 and turn 180 degrees around, what is to your left? A. Blue bin B. Black sofa", "images": ["imgA.png", "imgB.png"], "gt_answer": "B"}, # Expected: M1=front, M2=right. Turn 180. New Front=M_orig_behind (None), New Right = M_orig_left (None), New Left = M_orig_Right (M2). So left is M2 (black sofa).
            {"id": "translation_group000_q1", "category": [], "type": "forward", "meta_info": ["above", "objA", "objB", "objC"], "question": "Which direction is objA relative to objC? A. Above B. Below", "images": [], "gt_answer": "A"}
        ]
        # Create dummy file if it was the one missing
        if input_file == "/home/qineng/01_projects/03_mindcube_reasoning/other_all_image/qa/crossviewQA.jsonl":
             os.makedirs(os.path.dirname(input_file), exist_ok=True)
             with open(input_file, 'w', encoding='utf-8') as f_dummy:
                for entry in all_items:
                    f_dummy.write(json.dumps(entry) + '\n')

    rotation_items_processed_count = 0
    output_jsonl_items = []

    for item in all_items:
        item_id = item.get("id", "")
        if "rotation" in item_id:
            try:
                reasoning_chain = generate_rotation_reasoning_chain(item)
                item['reasoning_chain'] = reasoning_chain
                rotation_items_processed_count += 1
            except Exception as e:
                print(f"Error generating reasoning chain for item {item_id}: {e}")
                item['reasoning_chain'] = [f"Error generating reasoning chain for {item_id}: {str(e)}"]
            output_jsonl_items.append(item) # Add rotation items (processed or with error) to output
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item_to_write in output_jsonl_items:
            f.write(json.dumps(item_to_write, ensure_ascii=False) + '\n')
            
    print(f"Processed and added reasoning chains to {rotation_items_processed_count} 'rotation' QA pairs. Output saved to {output_file}")


if __name__ == "__main__":
    # Using the user-provided paths
    input_file_path = "/home/qineng/01_projects/03_mindcube_reasoning/other_all_image/qa/crossviewQA.jsonl"
    output_file_path = "/home/qineng/01_projects/03_mindcube_reasoning/other_all_image/qa/crossviewQA_rotation_with_reasoning.jsonl"
    
    add_reasoning_chains_to_data(input_file_path, output_file_path)