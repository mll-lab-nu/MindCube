import json
import re
import os
import os.path
import traceback

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
def generate_q1_reasoning_chain(question_text, meta_info, view_id, question_prefix, position,gt_answer_key):
    """
    Generate reasoning chain for q1 questions (self-movement direction).
    
    Args:
        images: List of image paths
        frame_mapping: Mapping from question frame to global frame
        main_objects: List of main objects in the scene
        other_objects: List of other objects in the scene
        answer_key: The correct answer key (A, B, etc.)
        answer_text: The text of the correct answer
        
    Returns:
        List of reasoning steps
    """
    reasoning = []
    obj_pool = meta_info[0]
    direction = meta_info[1]

    initial_viewpoint_idx = view_id  #1,2,3,4

    problem_id = position #1,2
    question = question_text

    # Start with a description of what we can see in each image without mentioning specific views
    reasoning.append("I need to determine how I moved from the viewpoint in image 1 to the viewpoint in image 2.")
    
    # Describe visible objects in each image using correct left-to-right ordering for each view

    reasoning.append(f"In image 1, I can see {obj_pool[0]} in front of the {obj_pool[(initial_viewpoint_idx + 2) % 4]}.")
    if problem_id==1:
        reasoning.append(f"In image 2, I can see {obj_pool[0]} in front of the {obj_pool[(initial_viewpoint_idx + 3) % 4]}.")
    else:
        reasoning.append(f"In image 2, I can see {obj_pool[0]} in front of the {obj_pool[(initial_viewpoint_idx + 1) % 4]}.")

 

    reasoning.append(f"I notice that {obj_pool[0]} is visible in both images, but from different angles.")
    
    # Detailed imagination-based reasoning about movement direction

    reasoning.append(f"I analyze how the viewpoint changed from image 1 to image 2 \
by analyzing how the structural features of objects on the platform and relative positions of these objects transform between the two views. \
This involves tracking how key features shift positions, observing which elements become more prominent or less visible, \
and comparing if these observed changes align with the expected spatial transformations that would occur when viewing the object from its left or right side.")

    if problem_id==1:
        reasoning.append(f"Image 2 seems to show the {obj_pool[0]}'s left side compared to the first one.")
        reasoning.append("This suggests that, to transition from the viewpoint in the first image to the viewpoint in the second image, I need to move forward and to the left.")


    else:
        reasoning.append(f"Image 2 seems to show the {obj_pool[0]}'s right side compared to the first one.")
        reasoning.append("This suggests that, to transition from the viewpoint in the first image to the viewpoint in the second image, I need to move forward and to the right.")

    options_map = extract_options_from_question(question)
    # Get the actual text for the correct answer letter
    correct_answer_text = options_map.get(gt_answer_key, "").strip()

    # Conclusion
    reasoning.append(f"Therefore, the answer is {gt_answer_key}. {correct_answer_text}")
    reasoning = " ".join(reasoning)
    return reasoning
def generate_no1_reasoning_chain(question_text, meta_info, view_id, question_prefix, position,gt_answer_key):

    
    reasoning = []
    obj_pool = meta_info[0]
    direction = meta_info[1]
    # 1. Scene Setup Observation: Describe views, primary objects, and camera movement

    object_observations_parts = []
    object_observations_parts.append(f"In image 1, I can see {obj_pool[0]} in front of the {obj_pool[3]}.")
    object_observations_parts.append(f"In image 2, I can see {obj_pool[0]} in front of the {obj_pool[4]}.")
    object_observations_parts.append(f"In image 3, I can see {obj_pool[0]} in front of the {obj_pool[1]}.")
    object_observations_parts.append(f"In image 4, I can see {obj_pool[0]} in front of the {obj_pool[2]}.")
    object_observations = " ".join(object_observations_parts)

    scene_description_intro = ""
    camera_movement_desc = ""

    scene_description_intro = f"In this scene, I observe four images showing different perspectives. All images feature the {obj_pool[0]} as the main object. {object_observations}"
    mental_process = "To identify the position change across views, I focus on the main object's angle variation. \
Then, I analyze the angles and relative positions of other objects on the platform to back up this observation. I understand that: " 
    camera_movement_desc = "Image 1 is the initial view. Image 2 is captured after a 90-degree clockwise rotation from image 1. Image 3 is after another 90-degree clockwise rotation (180 degrees from image 1). Image 4 is after a further 90-degree clockwise rotation (270 degrees from image 1)."
    
    reasoning.append(f"{scene_description_intro} {mental_process} {camera_movement_desc}")

    # 2. Question Interpretation: Parse viewpoint, turn, and target direction

    initial_viewpoint_idx = int(view_id)+1 if view_id!='gen'  else 'gen'  #1,2,3,4

    initial_problem_type = question_prefix #2,3,4,5,6
    problem_id = position #1,2
    question = question_text

    # base_layout_desc_parts = []

    # base_layout_desc_parts.append(f"In the first view (front view of {obj_pool[0]}), I can see that {obj_pool[3]} is behind the {obj_pool[0]}.")
    # base_layout_desc_parts.append(f"In the second view (left view of {obj_pool[0]}), I can see that {obj_pool[4]} is behind the {obj_pool[0]}.")
    # base_layout_desc_parts.append(f"In the third view (back view of {obj_pool[0]}), I can see that {obj_pool[1]} is behind the {obj_pool[0]}.")
    # base_layout_desc_parts.append(f"In the fourth view (right view of {obj_pool[0]}), I can see that {obj_pool[2]} is behind the {obj_pool[0]}.")
    # reasoning.append(f"{' '.join(base_layout_desc_parts)}.")

    mental_process2 =f"Through analyzing these perspective changes, I can construct a complete spatial understanding: when I view {obj_pool[4]} behind {obj_pool[0]} in the second view, it implies that in the first view, \
{obj_pool[4]} is on the right side of {obj_pool[0]}. Similarly, when I see {obj_pool[2]} behind {obj_pool[0]} in the fourth view, it indicates that in the first view, \
{obj_pool[2]} is on the left side of {obj_pool[0]}. \
However, I am still uncertain about what lies behind me in the first view. Then, I recognize that I can examine the opposite view to find out. \
The opposite view of the fist view is the third view. As {obj_pool[1]} is observed behind {obj_pool[0]} in the third view, it means that in the first view, \
{obj_pool[1]} is positioned behind me. This way, I can fully comprehend the spatial relationships of all objects in the entire scene."
    reasoning.append(mental_process2)
    
    base_layout_desc_parts = []

    # 3. P-O 1 + OO self
    # base_layout = determine_initial_object_layout(initial_viewpoint_idx, meta_info, scene_type)
    if (initial_problem_type==2 and problem_id==1) or initial_problem_type==5: 

        base_layout_desc_parts.append(f"{obj_pool[1:][(initial_viewpoint_idx -1 + 3) % 4]} is to the right of {obj_pool[0]}")
        base_layout_desc_parts.append(f"{obj_pool[1:][(initial_viewpoint_idx-1) % 4]} is to my behind")
        base_layout_desc_parts.append(f"{obj_pool[1:][(initial_viewpoint_idx -1 + 1) % 4]} is to the left of {obj_pool[0]}")
        reasoning.append(f"So, from the perspective of image {initial_viewpoint_idx}: {', '.join(base_layout_desc_parts)}.")

        # print(obj_pool)

        # print(reasoning)

    # 4. P-O 2

    if (initial_problem_type==2 and problem_id!=1) :
        if problem_id==2:
            turn_description = "left"
        elif problem_id==3:
            turn_description = "right"

        turned_layout_desc_parts = []
        if problem_id==2:
            turned_layout_desc_parts.append(f"{obj_pool[0]} is now to my right")
            turned_layout_desc_parts.append(f"{obj_pool[1:][(initial_viewpoint_idx -1 + 1) % 4]} is now to my front-right")
            reasoning.append(f"So, from the perspective of image {initial_viewpoint_idx}: after turning {turn_description}, {', '.join(turned_layout_desc_parts)}.")
            # print(obj_pool)
            reasoning.append(f" If I move forward, I wll be closer to the {obj_pool[1:][(initial_viewpoint_idx -1 + 1) % 4]}.")
        elif problem_id==3:
        
            turned_layout_desc_parts.append(f"{obj_pool[0]} is now to my left")
            turned_layout_desc_parts.append(f"{obj_pool[1:][(initial_viewpoint_idx -1 + 3) % 4]} is now to my front-left")

            reasoning.append(f"So, from the perspective of image {initial_viewpoint_idx}: after turning {turn_description}, {', '.join(turned_layout_desc_parts)}.")
            # print(obj_pool)
            reasoning.append(f" If I move forward, I wll be closer to the {obj_pool[1:][(initial_viewpoint_idx -1 + 3) % 4]}.")
        # print(reasoning)

    # 5. P-O perspective
    if initial_problem_type==3 or initial_problem_type==4:  
        if direction[0] == "left":
            front_object = obj_pool[2]
            behind_object = obj_pool[4]
            left_object = obj_pool[1]
            right_object = obj_pool[3]
            same_view = 4
        elif direction[0] == "face":
            front_object = obj_pool[1]
            behind_object = obj_pool[3]
            left_object = obj_pool[4]
            right_object = obj_pool[2]
            same_view = 3

        elif direction[0] == "back":
            front_object = obj_pool[3]
            behind_object = obj_pool[1]
            left_object = obj_pool[2]
            right_object = obj_pool[4]
            same_view = 1
        elif direction[0] == "right":
            front_object = obj_pool[4]
            behind_object = obj_pool[2]
            left_object = obj_pool[3]
            right_object = obj_pool[1]
            same_view = 2

        if initial_problem_type==3:
            reasoning.append(f"From the {obj_pool[0]}'s perspective, its front would be the same as the view {same_view} when facing the scene.")
            reasoning.append(f"Looking to the front from the {obj_pool[0]}'s position, the object I could see is the {front_object}; the object I couldn't see is the {behind_object}.")
        if initial_problem_type==4 : 
            reasoning.append(f"From the {obj_pool[0]}'s perspective, its front would be the same as the view {same_view} when facing the scene.")
            reasoning.append(f"From the {obj_pool[0]}'s position, the object to my left is the {left_object}; the object to my right is the {right_object}.")   
        # print(direction[0])
        # print(reasoning)

    # 6. O-O perspective

    if initial_problem_type==6 : 
        for i in range (1,len(direction)):

            if direction[i]!=None:
                # print(i)
                # print(obj_pool)
                # print(direction)
                if i==4:
                    left_object = obj_pool[1]
                    right_object = obj_pool[3]
                    same_view = 4
                elif i==3:
                    left_object = obj_pool[4]
                    right_object = obj_pool[2]
                    same_view = 3

                elif i==1:
                    left_object = obj_pool[2]
                    right_object = obj_pool[4]
                    same_view = 1
                elif i==2:
                    left_object = obj_pool[3]
                    right_object = obj_pool[1]
                    same_view = 2

                reasoning.append(f"From the {obj_pool[i]}'s perspective, its front would be the same as the view {same_view} when facing the scene.")
                if problem_id==1:
                    target_relative_direction = 'left'
                    reasoning.append(f"Looking to the front from the {obj_pool[i]}'s position, the object to the {target_relative_direction} of {obj_pool[0]} is the {left_object}.")

                if problem_id==2:
                    target_relative_direction = 'right'
                    reasoning.append(f"Looking to the front from the {obj_pool[i]}'s position, the object to the {target_relative_direction} of {obj_pool[0]} is the {right_object}.")


        # print(reasoning)



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
    # print(reasoning)
    
    return reasoning



def generate_reasoning_chain_among(item):
    """
    Generates a detailed reasoning chain for "around" type questions based on the item's data.
    This considers the specific objects visible in each view and their spatial relationships.
    """
    # try:
    question_text = item.get('question', '')
    answer_key = item.get('gt_answer', '')
    meta_info = item.get('meta_info', [])
    question_id = item.get('id', '')
    data_type = item.get('type')
    parts = question_id.split('_')
    view_id = int(parts[2][1]) if parts[2]!='gen' else 'gen'
    question_prefix = int(parts[3])
    position = int(parts[4])


    # Generate reasoning chain based on question type
    if question_prefix == 1:  # Self-movement direction

        reasoning_chain = generate_q1_reasoning_chain(question_text, meta_info, view_id, question_prefix, position, answer_key)
        
    else:
        reasoning_chain = generate_no1_reasoning_chain(question_text, meta_info, view_id, question_prefix, position, answer_key)
    
    return reasoning_chain
        
    # except Exception as e:
    #     # Return error chain with traceback
    #     error_msg = f"Error generating reasoning: {str(e)}"
    #     tb = traceback.format_exc()
    #     return [error_msg, f"Technical details: {tb}"]

def add_reasoning_chains_to_among_data(input_file, output_file):
    """
    Reads a JSONL file, adds reasoning chains to items with "around" in their ID,
    and writes ONLY these "around" items to a new JSONL file.
    """
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    all_items_from_input = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                if line.strip():
                    all_items_from_input.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from input file {input_file}. Details: {e}")
        return

    around_items_processed_count = 0
    error_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item_idx, item in enumerate(all_items_from_input):
            # try:
                item_id = item.get("id", "")
                if "among" in item_id: # Process only items with "around" in their ID
                    reasoning_chain = generate_reasoning_chain_among(item)
                    item['reasoning_chain'] = reasoning_chain
                    # f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                    around_items_processed_count += 1
            # except Exception as e:
            #     error_count += 1
            #     print(f"Error processing item at index {item_idx} with ID '{item.get('id', 'N/A')}': {e}")
            #     # Optionally handle items that cause errors during reasoning generation

                    f_out.write(json.dumps(item, ensure_ascii=False) + '\n') # Write errored items
                    around_items_processed_count += 1


    print(f"Processed and saved {around_items_processed_count} 'around' items to {output_file}")
    print(f"Encountered {error_count} errors during processing")

if __name__ == "__main__":
    # Using the user-provided paths
    input_file_path = r"C:\Users\FS139\Desktop\qa_release\crossviewQA_train_shuffled.jsonl"
    output_file_path = r"C:\Users\FS139\Desktop\qa_release\crossviewQA_among_with_reasoning.jsonl"
    add_reasoning_chains_to_among_data(input_file_path, output_file_path)