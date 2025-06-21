import json
import re
import os

def generate_reasoning_chain(question, answer, category, meta_info, question_type=None):
    """
    Generate a reasoning chain that uses meta_info to perform transitive reasoning.
    
    Args:
        question (str): The question text
        answer (str): The ground truth answer (A, B, C, D)
        category (list): Category information
        meta_info (list): Meta information [spatial_relation, obj1, obj2, obj3, ...]
        question_type (str): Type of reasoning (forward or inverse)
    
    Returns:
        list: A reasoning chain showing transitive reasoning
    """
    reasoning = []
    
    # Extract objects from question and meta_info
    question_objs = extract_objects_from_question(question)
    
    # Parse meta_info to get spatial relation and objects
    if len(meta_info) >= 3:
        spatial_relation = meta_info[0]
        objects = meta_info[1:] if len(meta_info) > 1 else []
    else:
        spatial_relation = ""
        objects = []
    
    # Extract the objects we're asking about from the question
    target_obj1, target_obj2 = question_objs
    
    # Parse options from question
    options = question.split("?")[1] if "?" in question else ""
    option_texts = {}
    
    # Extract option labels - fix to properly extract individual options
    for opt in ["A", "B", "C", "D"]:
        match = re.search(rf"{opt}\.\s+([A-Za-z ]+)(?=\s+[A-D]\.|$)", options)
        if match:
            option_texts[opt] = match.group(1).strip()
    
    # Get correct answer text - ensure we only get the specific option
    answer_text = option_texts.get(answer, "")
    
    # Determine if this is inverse reasoning (images swapped)
    is_view_inverse = question_type == "inverse reasoning" if question_type else False
    
    # Map spatial relations to natural language
    relation_map = {
        "left": "to the left of",
        "right": "to the right of",
        "on": "above",
        "above": "above",
        "down": "below",
        "below": "below",
        "front": "in front of",
        "behind": "behind"
    }
    
    # For inverse relations, determine the correct relationship
    inverse_map = {
        "left": "right",
        "right": "left",
        "on": "down",
        "above": "below",
        "down": "on",
        "below": "above",
        "front": "behind",
        "behind": "front"
    }
    
    # Handle different types of reasoning based on meta_info
    if len(objects) >= 3:
        # For typical meta_info: [relation, obj1, obj2, obj3]
        middle_obj = objects[1]  # Usually the linking object
        
        # Handle complex spatial relations (e.g., "front,down")
        if isinstance(spatial_relation, str) and "," in spatial_relation:
            # Split the compound relation into separate relations
            relations = spatial_relation.split(",")
            relation1 = relations[0]
            relation2 = relations[1] if len(relations) > 1 else relation1
            
            # Get natural language descriptions
            relation1_desc = relation_map.get(relation1, relation1)
            relation2_desc = relation_map.get(relation2, relation2)
            
            # Build reasoning chain for compound relations
            # Adjust view order based on is_view_inverse, but keep names consistent
            if is_view_inverse:
                # For inverse view order (second image is shown first)
                reasoning.append(f"In the first view (view 1), I can see that {middle_obj} is {relation2_desc} {objects[2]}.")
                reasoning.append(f"In the second view (view 2), I can see that {objects[0]} is {relation1_desc} {middle_obj}.")
            else:
                # For normal view order
                reasoning.append(f"In the first view (view 1), I can see that {objects[0]} is {relation1_desc} {middle_obj}.")
                reasoning.append(f"In the second view (view 2), I can see that {middle_obj} is {relation2_desc} {objects[2]}.")
            
            reasoning.append(f"I can identify that the {middle_obj} appearing in both views is the same object, which allows me to connect these observations.")
            reasoning.append("Combining these observations through spatial reasoning:")
            
            # Determine if question asks about objects in reverse order
            # If target_obj1 = objects[2] and target_obj2 = objects[0], then it's a reverse question
            is_reverse_question = (target_obj1 == objects[2] and target_obj2 == objects[0])
            
            if is_reverse_question:
                # Get inverse relations
                inv_relation1 = inverse_map.get(relation1, "")
                inv_relation2 = inverse_map.get(relation2, "")
                inv_relation1_desc = relation_map.get(inv_relation1, inv_relation1)
                inv_relation2_desc = relation_map.get(inv_relation2, inv_relation2)
                
                reasoning.append(f"The question asks about how {objects[2]} relates to {objects[0]}, which requires the inverse relation.")
                
                # Determine which relation matches the answer
                target_relation = ""
                if answer_text.lower() in ["left", "right"]:
                    target_relation = inv_relation1 if inv_relation1 in ["left", "right"] else inv_relation2
                elif answer_text.lower() in ["above", "below"]:
                    target_relation = inv_relation1 if inv_relation1 in ["on", "down", "above", "below"] else inv_relation2
                elif answer_text.lower() in ["in front", "behind"]:
                    target_relation = inv_relation1 if inv_relation1 in ["front", "behind"] else inv_relation2
                
                if target_relation:
                    target_relation_desc = relation_map.get(target_relation, target_relation)
                    reasoning.append(f"Based on the transitive relationship through {middle_obj}, I conclude that {objects[2]} is {target_relation_desc} {objects[0]}.")
            else:
                # Standard question (asking about objects[0] relative to objects[2])
                # Determine which relation matches the answer
                target_relation = ""
                if answer_text.lower() in ["left", "right"]:
                    target_relation = relation1 if relation1 in ["left", "right"] else relation2
                elif answer_text.lower() in ["above", "below"]:
                    target_relation = relation1 if relation1 in ["on", "down", "above", "below"] else relation2
                elif answer_text.lower() in ["in front", "behind"]:
                    target_relation = relation1 if relation1 in ["front", "behind"] else relation2
                
                if target_relation:
                    target_relation_desc = relation_map.get(target_relation, target_relation)
                    reasoning.append(f"Based on the transitive relationship through {middle_obj}, I conclude that {objects[0]} is {target_relation_desc} {objects[2]}.")
        else:
            # Handle single relation
            relation_desc = relation_map.get(spatial_relation, spatial_relation)
            
            # Build reasoning chain based on view order, but keep names consistent
            if is_view_inverse:
                # For inverse view order (second image is shown first)
                reasoning.append(f"In the first view (view 1), I can see that {middle_obj} is {relation_desc} {objects[2]}.")
                reasoning.append(f"In the second view (view 2), I can see that {objects[0]} is {relation_desc} {middle_obj}.")
            else:
                # For normal view order
                reasoning.append(f"In the first view (view 1), I can see that {objects[0]} is {relation_desc} {middle_obj}.")
                reasoning.append(f"In the second view (view 2), I can see that {middle_obj} is {relation_desc} {objects[2]}.")
            
            reasoning.append(f"I can identify that the {middle_obj} appearing in both views is the same object, which allows me to connect these observations.")
            reasoning.append(f"Combining these observations through transitive reasoning:")
            reasoning.append(f"If {objects[0]} is {relation_desc} {middle_obj}, and {middle_obj} is {relation_desc} {objects[2]},")
            reasoning.append(f"then {objects[0]} is {relation_desc} {objects[2]}.")
            
            # Determine if question asks about objects in reverse order
            is_reverse_question = (target_obj1 == objects[2] and target_obj2 == objects[0])
            
            if is_reverse_question:
                # For reverse question, we need the inverse relation
                inv_relation = inverse_map.get(spatial_relation, "")
                inv_relation_desc = relation_map.get(inv_relation, inv_relation)
                
                reasoning.append(f"The question asks about {objects[2]} relative to {objects[0]}, which requires the inverse relation.")
                reasoning.append(f"If {objects[0]} is {relation_desc} {objects[2]}, then {objects[2]} must be {inv_relation_desc} {objects[0]}.")
    
    # Add conclusion with specific answer
    reasoning.append(f"The answer is {answer}. {answer_text}")
    
    return reasoning


def extract_objects_from_question(question):
    """Extract object names from the question using regex pattern matching"""
    # Pattern for "which direction is the X relative to the Y?"
    pattern = r"[wW]hich direction is the ([a-zA-Z0-9 ]+) relative to the ([a-zA-Z0-9 ]+)\?"
    match = re.search(pattern, question, re.IGNORECASE)
    
    if match:
        return match.group(1).strip(), match.group(2).strip()
    
    # Fallback to generic object names
    return "first object", "second object"


def add_reasoning_chains_to_data(input_file, output_file):
    """
    Read a JSONL file with QA pairs and add reasoning chains to each entry
    
    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output JSONL file with reasoning chains added
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Read input file
    qa_items = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                qa_items.append(json.loads(line))
    
    # Filter items to only include those with "translation" in the ID
    translation_items = [item for item in qa_items if "translation" in item.get("id", "")]
    print(f"Found {len(translation_items)} items with 'translation' in ID out of {len(qa_items)} total items")
    
    # Add reasoning chains
    for item in translation_items:
        question = item.get('question', '')
        answer = item.get('gt_answer', '')
        category = item.get('category', [])
        meta_info = item.get('meta_info', [])
        question_type = item.get('type', None)
        
        # Generate reasoning chain
        reasoning_chain = generate_reasoning_chain(
            question=question,
            answer=answer,
            category=category,
            meta_info=meta_info,
            question_type=question_type
        )
        
        # Add to item
        item['reasoning_chain'] = reasoning_chain
    
    # Write output file with all items (only translation items have reasoning chains)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in qa_items:
            if "translation" in item.get("id", ""):
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Added reasoning chains to {len(translation_items)} QA pairs and saved to {output_file}")


if __name__ == "__main__":
    # Use the provided paths
    input_file = "/home/qineng/01_projects/03_mindcube_reasoning/other_all_image/qa/crossviewQA.jsonl"
    output_file = "/home/qineng/01_projects/03_mindcube_reasoning/other_all_image/qa/crossviewQA_linear_with_reasoning.jsonl"
    
    # Process the file
    add_reasoning_chains_to_data(input_file, output_file)