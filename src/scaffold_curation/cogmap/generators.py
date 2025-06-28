"""
Cognitive Map Generators

This module contains the core cognitive map generation logic extracted from 
the original grounded_cogmap_gen.py with minimal changes to preserve functionality.

Main generators for different spatial reasoning settings:
- Around: Object position relative to viewpoints
- Among: Objects distributed among viewpoints  
- Translation: Object movement scenarios
- Rotation: Object rotation scenarios
"""

import json
import os
import re
from typing import Tuple, Dict, List, Union, Optional
from ...utils import load_jsonl, save_jsonl, normalize_direction


# Constants from original file (preserved unchanged)
COG_MAP_DESCRIPTION_FOR_INPUT_SEP_OBJ_VIEW_OLD = '''[Cognitive Map Format]
We provide you a 2D grid map of the scene that is related to the question you should answer. Below is the description of the map:
- The map uses a 10x10 grid where [0,0] is at the top-left corner and [9,9] is at the bottom-right corner {birdview}
- Directions are defined as:
  * up = towards the top of the grid (decreasing y-value)
  * right = towards the right of the grid (increasing x-value)
  * down = towards the bottom of the grid (increasing y-value)
  * left = towards the left of the grid (decreasing x-value)
  * inner = straight into the 2D map (perpendicular to the grid, pointing away from you)
  * outer = straight out of the 2D map (perpendicular to the grid, pointing towards you)
- "objects" lists all important items in the scene with their positions
- "facing" indicates which direction an object is oriented towards (when applicable)
- "views" represents the different camera viewpoints in the scene
'''

COG_MAP_DESCRIPTION_FOR_INPUT = '''[Cognitive Map Format]
We provide you a 2D grid map of the scene that is related to the question you should answer. Below is the description of the map:
- The map uses a 10x10 grid where [0,0] is at the top-left corner and [9,9] is at the bottom-right corner {birdview}
- Directions are defined as:
  * up = towards the top of the grid (decreasing y-value)
  * right = towards the right of the grid (increasing x-value)
  * down = towards the bottom of the grid (increasing y-value)
  * left = towards the left of the grid (decreasing x-value)
  * inner = straight into the 2D map (perpendicular to the grid, pointing away from you)
  * outer = straight out of the 2D map (perpendicular to the grid, pointing towards you)
- "objects" lists all important items in the scene with their positions
- "facing" indicates which direction an object is oriented towards (when applicable)
- "views" represents the different camera viewpoints in the scene
'''

COG_MAP_DESCRIPTION_FOR_OUTPUT = '''[Task]
Your task is to analyze the spatial arrangement of objects in the scene by examining the provided images, which show the scene from different viewpoints. You will then create a detailed cognitive map representing the scene using a 10x10 grid coordinate system. 
[Rules]
1. Focus ONLY on these categories of objects in the scene: {{obj}}
2. Create a cognitive map with the following structure{birdview}:
   - A 10x10 grid where [0,0] is at the top-left corner and [9,9] is at the bottom-right corner
   - up = towards the top of the grid (decreasing y)
   - right = towards the right of the grid (increasing x)
   - down = towards the bottom of the grid (increasing y)
   - left = towards the left of the grid (decreasing x)
   - inner = straight into the 2D map (perpendicular to the grid, pointing away from you)
   - outer = straight out of the 2D map (perpendicular to the grid, pointing towards you)
   - Include positions of all objects from the specified categories
   - Estimate the center location (coordinates [x, y]) of each instance within provided categories
   - If a category contains multiple instances, include all of them
   - Each object's estimated location should accurately reflect its real position in the scene, preserving the relative spatial relationships among all objects
   - Combine and merge information from the images since they are pointing to the same scene, calibrating the object locations accordingly
   - Include camera positions and directions for each view
3. Carefully integrate information from all views to create a single coherent spatial representation.
<orientation_info>'''

COG_MAP_DESCRIPTION_FOR_OUTPUT_SHORTEN = '''[Task]
Your task is to analyze the spatial arrangement of objects in the scene by examining the provided images, which show the scene from different viewpoints. You will then create a detailed cognitive map representing the scene using a **10x10 grid coordinate system**. 
[Rules]
1. Focus ONLY on these categories of objects in the scene: {{obj}}
2. Create a cognitive map with the following structure{birdview}:
   - A 10x10 grid where [0, 0] is at the top-left corner and [9, 9] is at the bottom-right corner
   - up = towards the top of the grid (decreasing y)
   - right = towards the right of the grid (increasing x)
   - down = towards the bottom of the grid (increasing y)
   - left = towards the left of the grid (decreasing x)
   - Include positions of all objects from the specified categories
   - Estimate the center location (coordinates [x, y]) of each instance within provided categories
   - If a category contains multiple instances, include all of them
   - Object positions must maintain accurate relative spatial relationships
   - Combine and merge information from the images since they are pointing to the same scene, calibrating the object locations with grid coordinates accordingly
3. Carefully integrate information from all views to create a single coherent spatial representation.
<orientation_info>'''

# =====[06-27-2025 Patch] ======================================================
# Due to two around image groups do not comply with the rules, we need to add a special case for them.
# Will remove this patch after the dataset is fixed.
def special_around_id_list_062725() -> List[str]:
    return [
        "d254259ad5be1ec58a5591aa18011ee766c05bf76fd1ea4cd17fe994e6e7707c",
        "e3f7e530666356f04f6ea50d7eb3173c130d7d5b0da8bcb332de4cf20961c60c"
    ]
    
def id_is_in_special_around_id_list_062725(id: str) -> bool:
    for sid in special_around_id_list_062725():
        if sid in id:
            return True
    return False
# ==============================================================================


def format_cogmap_json(cogmap):
    """
    Format a dictionary into a specific JSON string format.
    
    Args:
        cogmap: Dictionary with 'objects' and 'views' keys
        
    Returns:
        str: Formatted JSON string
    """
    result = "{\n"
    result += '  "objects": [\n'
    for i, obj in enumerate(cogmap["objects"]):
        result += '    ' + json.dumps(obj, ensure_ascii=False)
        if i < len(cogmap["objects"]) - 1:
            result += ','
        result += '\n'
    result += '  ],\n'
    
    result += '  "views": [\n'
    for i, view in enumerate(cogmap["views"]):
        result += '    ' + json.dumps(view, ensure_ascii=False)
        if i < len(cogmap["views"]) - 1:
            result += ','
        result += '\n'
    result += '  ]\n'
    result += '}'
    
    # Validate that the formatted JSON is equivalent to the original
    try:
        parsed_json = json.loads(result)
        # Check if the parsed JSON is equivalent to the original dictionary
        if parsed_json["objects"] != cogmap["objects"] or parsed_json["views"] != cogmap["views"]:
            print(f"Warning: Formatted JSON is not equivalent to the original dictionary")
            print(f"Original: {json.dumps(cogmap, indent=2, ensure_ascii=False, separators=(',', ': '))}")
            print(f"Formatted and parsed: {json.dumps(parsed_json, indent=2, ensure_ascii=False, separators=(',', ': '))}")
            raise ValueError("Formatted JSON is not equivalent to the original dictionary")
    except json.JSONDecodeError as e:
        print(f"Warning: Generated JSON is not valid: {e}")
        print(f"Original JSON: {json.dumps(cogmap, indent=2, ensure_ascii=False, separators=(',', ': '))}")
        raise e
    
    return result


def extract_around_image_names(images: List[str]) -> List[int]:
    """
    Extract the image names from the images list.
    """
    image_names = []
    for image in images:
        try:
            filename = os.path.basename(image)
        except Exception:
            filename = image
        match = re.search(r'(\d+)_frame(?:_[^.]+)?\.(?:png|jpg|jpeg)', filename)
        if match:
            ### special case for dl3dv10k
            num = int(match.group(1))
            if num == 33: # a typo in the dataset, changing it will cost a lot of time, so we just change it here
                num = 3
            image_names.append(num)
        else:
            raise ValueError(f"Error extracting image name from {image}")
    return image_names


def gen_object_coordinate_dict(name, position, facing=None):
    if name == "":
        return None
    if facing is None:
        return {"name": name, "position": position}
    else:
        return {"name": name, "position": position, "facing": facing}


def generate_around_cogmap(item) -> Tuple[str, list, list]:
    """
    **Now this is the augmented version of the around setting, including the object orientation.**
    Generate the cogmap for around setting.
    return the cogmap into three parts:
     - the first part is the description of the cogmap, which is a string.
     - the second part is the cogmap, which is a dict string.
     - the third part is the objects that have None orientation, which is a list of dicts.
    """
    id = item.get("id", "")
    category = item.get("category", [])
    type = item.get("type", -1)
    meta_info = item.get("meta_info", [])
    question = item.get("question", "")
    question_images = item.get("images", [])
    gt_answer = item.get("answer", "")

    datasource = id.split("_")[0].replace("around", "")
    if datasource == "new":
        datasource = "self"
    else:
        datasource = "dl3dv10k"

    assert datasource in ["self", "dl3dv10k"], f"Unknown datasource: {datasource}"

    image_group_num = meta_info[0][0] # how many views in the entire image group
    object_len = meta_info[1][0]
    objects = meta_info[1][1] # the objects in the scene
    objects_orientation = meta_info[1][2] # orientation info of the objects

    orientation_mapping = {
        "face": "down",
        "back": "up",
        "right": "right",
        "left": "left",
        "None": None,
        "null": None,
        None: None,
    }

    assert object_len == len(objects), f"Object number {object_len} is not equal to objects length {len(objects)}, id: {id}"
    assert object_len == len(objects_orientation), f"Object number {object_len} is not equal to objects orientation length {len(objects_orientation)}, id: {id}"
    question_images_len = len(question_images) # how many images in the question
    assert isinstance(image_group_num, int), f"Image group number {image_group_num} is not an integer, id: {id}"
    assert isinstance(question_images_len, int), f"Question images length {question_images_len} is not an integer, id: {id}"
    assert question_images_len <= image_group_num, f"Question images length {question_images_len} is greater than image group number {image_group_num}, id: {id}"
    assert 3 <= image_group_num <= 6, f"Image group number {image_group_num} is not between 3 and 6"

    if id_is_in_special_around_id_list_062725(id): # [patch 06-27-2025], should be removed after the dataset is fixed.
        global_views = {1: "front", 2: "left", 3: "right", 4: "front", 5: "left", 6: "right"}
    elif image_group_num == 3 and datasource == "self":
        global_views = {1: "front", 2: "left", 3: "right"}
    elif image_group_num == 3 and datasource == "dl3dv10k":
        global_views = {1: "front", 2: "left", 3: "right", 4: "back"}
    elif image_group_num == 4:
        global_views = {1: "front", 2: "left", 3: "right", 4: "back"}
    elif image_group_num == 5:
        global_views = {1: "front", 2: "left", 3: "right", 4: "left", 5: "right"}
    elif image_group_num == 6 and datasource == "self":
        global_views = {1: "front", 2: "left", 3: "right", 4: "left", 5: "right", 6: "back"}
    elif image_group_num == 6 and datasource == "dl3dv10k":
        global_views = {1: "front", 2: "left", 3: "right", 4: "front", 5: "left", 6: "right"}
    else:
        raise ValueError(f"Unknown image group number: {image_group_num}, datasource: {datasource}")

    question_image_ids = extract_around_image_names(question_images)

    assert len(question_image_ids) == question_images_len, f"Question image ids length {len(question_image_ids)} is not equal to question images length {question_images_len}"
    # Ensure no repeated numbers in question_image_ids
    assert len(question_image_ids) == len(set(question_image_ids)), f"Duplicate image IDs found in question_image_ids: {question_image_ids}"
    # Validate that each image ID is within the valid range for this group
    for img_id in question_image_ids:
        assert 1 <= img_id <= len(global_views.items()), f"Image ID {img_id} is outside the valid range [1, {len(global_views.items())}]. question id: {id}"

    view_base_name = "Image" if "image" in question.lower() else "View"

    local_view_map_to_global_view_image_id = [
        (f"{view_base_name} {k+1}", v) for k, v in enumerate(question_image_ids)
    ] # [(View 1, g1), (View 2, g2), ...], g means global view image id

    # we have built the mapping relationship. now it is time to determine all the posibilities of the object positions.
    ## how to do this? well, we still can first determine the object arragement first, which is easy
    ## then we can determine the view positions. this one is a little tricky, and it is where we need to use our mapping relationship.

    # mapping to 2, 3, 4 objects coordinates
    mapping_view_to_coordinates = {
        "front": [[5, 6], [5, 6], [5, 6]], # facing up
        "left": [[3, 5], [3, 5], [2, 5]], # facing right
        "right": [[6, 5], [7, 5], [7, 5]], # facing left
        "back": [[5, 4], [5, 4], [5, 4]] # facing down
    }

    facing_mapping = {
        "front": "up",
        "left": "right",
        "right": "left",
        "back": "down"
    }

    ### for object, {2, 3, 4}. meaning 2 to 4 objects
    if object_len == 2: # (4, 5), (5, 5)
        object_coordinates = [
            gen_object_coordinate_dict(objects[0], [4, 5], orientation_mapping[objects_orientation[0]]), # eg. objects_orientation[0] = "face", orientation_mapping["face"] = "down"
            gen_object_coordinate_dict(objects[1], [5, 5], orientation_mapping[objects_orientation[1]])
        ]
    elif object_len == 3: # (4, 5), (5, 5), (6, 5)
        object_coordinates = [
            gen_object_coordinate_dict(objects[0], [4, 5], orientation_mapping[objects_orientation[0]]),
            gen_object_coordinate_dict(objects[1], [5, 5], orientation_mapping[objects_orientation[1]]),
            gen_object_coordinate_dict(objects[2], [6, 5], orientation_mapping[objects_orientation[2]])
        ]
    elif object_len == 4: # (3, 5), (4, 5), (5, 5), (6, 5)
        object_coordinates = [
            gen_object_coordinate_dict(objects[0], [3, 5], orientation_mapping[objects_orientation[0]]),
            gen_object_coordinate_dict(objects[1], [4, 5], orientation_mapping[objects_orientation[1]]),
            gen_object_coordinate_dict(objects[2], [5, 5], orientation_mapping[objects_orientation[2]]),
            gen_object_coordinate_dict(objects[3], [6, 5], orientation_mapping[objects_orientation[3]])
        ]
    
    mapping_index = object_len - 2 # eg. object_len = 2, mapping_index = 0
    view_coordinates = []
    for local_view_name, global_id in local_view_map_to_global_view_image_id:
        global_view = global_views[global_id] # eg. global_view = "front"
        view_coordinate = mapping_view_to_coordinates[global_view][mapping_index] # eg. global_view = "front", mapping_index = 0, view_coordinate = [5, 6]
        view_coordinates.append({
            "name": local_view_name, # eg. local_view_name = "View 1"
            "position": view_coordinate, # eg. view_coordinate = [5, 6]
            "facing": facing_mapping[global_view] # eg. facing_mapping["front"] = "up"
        })
    # filter out the objects that is None
    object_coordinates = [obj for obj in object_coordinates if obj is not None]
    cogmap = {
        "objects": object_coordinates,
        "views": view_coordinates
    }
    
    # gather objects that have None orientation and store them in a list
    oriented_objects = [obj['name'] for obj in object_coordinates if 'facing' in obj.keys()]
    return format_cogmap_json(cogmap), objects, oriented_objects


def generate_among_cogmap(item) -> Tuple[str, list, list]:
    """
    **Now this is the augmented version of the among setting, including the object orientation.**
    Generate the cogmap for among setting.
    """
    id = item.get("id", "")
    category = item.get("category", [])
    type = item.get("type", "")
    objects = item.get("meta_info", [])[0] # list of objects
    objects_orientation = item.get("meta_info", [])[1] # list of objects orientation
    question = item.get("question", "")
    images = item.get("images", [])

    assert len(objects) == 5, f"Among setting should have 5 objects, but got {len(objects)}, id: {id}"
    assert len(objects_orientation) == 5, f"Among setting should have 5 objects orientation, but got {len(objects_orientation)}"
    assert len(images) == 2 or len(images) == 4, f"Among setting should have 2 or 4 images, but got {len(images)}"

    orientation_mapping = {
        "face": "down",
        "back": "up",
        "right": "right",
        "left": "left",
        "None": None,
        "null": None,
        None: None,
    }

    image_names = [os.path.basename(image) for image in images]
    image_names = [name.split("_")[0] for name in image_names]

    view_base_name = "Image" if "image" in question.lower() else "View"

    # Check if all image names are recognizable direction names
    recognizable_directions = ["front", "left", "right", "back"]
    all_recognizable = all(image_name in recognizable_directions for image_name in image_names)
    
    if all_recognizable:
        # All images have recognizable direction names, process normally
        processed_image_names = image_names
    else:
        # Some images don't have recognizable direction names
        if len(images) != 4:
            raise ValueError(f"Non-recognizable image naming detected for item {id}. "
                           f"Image names: {image_names}. "
                           f"For non-direction naming, exactly 4 images are required, but got {len(images)}")
        
        # Use fallback mapping for exactly 4 images with non-recognizable names
        default_directions = ["front", "left", "back", "right"]
        processed_image_names = default_directions[:len(image_names)]
        print(f"Warning: Using fallback direction mapping for item {id}. Image names: {image_names} -> {processed_image_names}")
    
    # Validate that all processed image names are valid directions
    for image_name in processed_image_names:
        assert image_name in ["front", "left", "right", "back"], f"Unknown image name: {image_name}"
    
    local_view_map_to_global_view = [
        (f"{view_base_name} {k+1}", v) for k, v in enumerate(processed_image_names)
    ] # here, v is the global view image name, eg. v = "front"

    mapping_view_to_coordinates = {
        "front": [5, 6],
        "left": [4, 5],
        "right": [6, 5],
        "back": [5, 4]
    }

    facing_mapping = {
        "front": "up",
        "left": "right",
        "right": "left",
        "back": "down"
    }

    object_coordinates = [
        gen_object_coordinate_dict(objects[0], [5, 5], orientation_mapping[objects_orientation[0]]),
        gen_object_coordinate_dict(objects[1], [5, 8], orientation_mapping[objects_orientation[1]]),
        gen_object_coordinate_dict(objects[2], [2, 5], orientation_mapping[objects_orientation[2]]),
        gen_object_coordinate_dict(objects[3], [5, 2], orientation_mapping[objects_orientation[3]]),
        gen_object_coordinate_dict(objects[4], [8, 5], orientation_mapping[objects_orientation[4]])
    ]
    
    # extract the image names
    view_coordinates = []
    for local_view_name, global_view_name in local_view_map_to_global_view: # eg. local_view_name = "View 1", global_view_name = "front"
        view_coordinate = mapping_view_to_coordinates[global_view_name] # eg. mapping_view_to_coordinates["front"] = [5, 4]
        view_facing = facing_mapping[global_view_name] # eg. facing_mapping["front"] = "up"
        view_coordinates.append({
            "name": local_view_name, # eg. local_view_name = "View 1"
            "position": view_coordinate, # eg. view_coordinate = [5, 4]
            "facing": view_facing # eg. view_facing = "up"
        })
    # filter out the objects that is None
    object_coordinates = [obj for obj in object_coordinates if obj is not None]

    cogmap = {
        "objects": object_coordinates,
        "views": view_coordinates
    }
    oriented_objects = [obj['name'] for obj in object_coordinates if 'facing' in obj.keys()]
    return format_cogmap_json(cogmap), objects, oriented_objects


def generate_translation_cogmap(item) -> Tuple[str, list, list]:
    """
    Generate the cogmap for translation setting.
    """
    id = item.get("id", "")
    category = item.get("category", [])
    type = item.get("type", "")
    meta_info = item.get("meta_info", [])
    question = item.get("question", "")
    images = item.get("images", [])
    gt_answer = item.get("answer", "")

    spatial_relations = meta_info[0]
    spatial_relation_list = spatial_relations.split(",")
    relation_1 = spatial_relation_list[0]
    relation_2 = spatial_relation_list[1] if len(spatial_relation_list) > 1 else relation_1

    #TODO: generate the cogmap
    # handle different types of spatial relationships
    # center is (5, 5), horizontal is x, vertical is y
    objects = meta_info[1:] # obj 1 is down to obj 2, obj 2 is down to obj 3
    if (relation_1, relation_2) == ('down', 'down'):
        # change y only
        objects_coordnates = [
            gen_object_coordinate_dict(objects[0], [5, 7]),
            gen_object_coordinate_dict(objects[1], [5, 5]), # obj 1 is down to obj 2
            gen_object_coordinate_dict(objects[2], [5, 3]) # obj 2 is down to obj 3
        ]
        view_0 = {"position": [5, 6], "facing": "inner"} # x
        view_1 = {"position": [5, 4], "facing": "inner"} # x
    elif (relation_1, relation_2) == ('right', 'right'):
        # change x only
        objects_coordnates = [
            gen_object_coordinate_dict(objects[0], [7, 5]),
            gen_object_coordinate_dict(objects[1], [5, 5]), # obj 1 is right to obj 2
            gen_object_coordinate_dict(objects[2], [3, 5]) # obj 2 is right to obj 3
        ]
        view_0 = {"position": [6, 6], "facing": "up"}
        view_1 = {"position": [4, 6], "facing": "up"}
    elif (relation_1, relation_2) == ('left', 'left'):
        # change x only
        objects_coordnates = [
            gen_object_coordinate_dict(objects[0], [3, 5]),
            gen_object_coordinate_dict(objects[1], [5, 5]), # obj 1 is left to obj 2
            gen_object_coordinate_dict(objects[2], [7, 5]) # obj 2 is left to obj 3
        ]
        view_0 = {"position": [4, 6], "facing": "up"} # ⬆️
        view_1 = {"position": [6, 6], "facing": "up"} # ⬆️
    elif (relation_1, relation_2) == ('front', 'down'):
        # change both x and y
        objects_coordnates = [
            gen_object_coordinate_dict(objects[0], [7, 5]),
            gen_object_coordinate_dict(objects[1], [5, 5]), # obj 1 is front to obj 2
            gen_object_coordinate_dict(objects[2], [5, 3]) # obj 2 is down to obj 3
        ]
        view_0 = {"position": [8, 5], "facing": "left"} # ⬅️
        view_1 = {"position": [5, 4], "facing": "inner"} # x
    elif (relation_1, relation_2) == ('right', 'down'):
        # change both x and y
        objects_coordnates = [
            gen_object_coordinate_dict(objects[0], [7, 5]),
            gen_object_coordinate_dict(objects[1], [5, 5]), # obj 1 is right to obj 2
            gen_object_coordinate_dict(objects[2], [5, 3]) # obj 2 is down to obj 3
        ]
        view_0 = {"position": [6, 6], "facing": "up"} # ⬆️
        view_1 = {"position": [5, 4], "facing": "inner"} # x
    elif (relation_1, relation_2) == ('front', 'front'):
        # change y only
        objects_coordnates = [
            gen_object_coordinate_dict(objects[0], [5, 7]),
            gen_object_coordinate_dict(objects[1], [5, 5]),
            gen_object_coordinate_dict(objects[2], [5, 3])
        ]
        view_0 = {"position": [5, 8], "facing": "up"}
        view_1 = {"position": [5, 6], "facing": "up"}
    elif (relation_1, relation_2) == ('on', 'behind'):
        # change both x and y
        objects_coordnates = [
            gen_object_coordinate_dict(objects[0], [5, 3]),
            gen_object_coordinate_dict(objects[1], [5, 5]), # obj 1 is on obj 2
            gen_object_coordinate_dict(objects[2], [3, 5]) # obj 2 is behind obj 3
        ]
        view_0 = {"position": [5, 4], "facing": "inner"} # x
        view_1 = {"position": [6, 5], "facing": "left"} # x
    elif (relation_1, relation_2) == ('on', 'on'):
        # change y only
        objects_coordnates = [
            gen_object_coordinate_dict(objects[0], [5, 3]),
            gen_object_coordinate_dict(objects[1], [5, 5]),
            gen_object_coordinate_dict(objects[2], [5, 7])
        ]
        view_0 = {"position": [5, 6], "facing": "inner"}
        view_1 = {"position": [5, 4], "facing": "inner"}
    else:
        raise ValueError(f"Unknown spatial relation: {relation_1}, {relation_2}")
    view_0_name_dict = {"name": "View 1"}
    view_1_name_dict = {"name": "View 2"}
    view_coordinates = []
    if 'inverse' in type:
        view_0_name_dict.update(view_1)
        view_1_name_dict.update(view_0)
        view_coordinates.append(view_0_name_dict)
        view_coordinates.append(view_1_name_dict)
    else:
        view_0_name_dict.update(view_0)
        view_1_name_dict.update(view_1)
        view_coordinates.append(view_0_name_dict)
        view_coordinates.append(view_1_name_dict)
    # filter out the objects that is None
    objects_coordnates = [obj for obj in objects_coordnates if obj is not None]
    cogmap = {
        "objects": objects_coordnates,
        "views": view_coordinates
    }   
    return format_cogmap_json(cogmap), objects, []


def generate_rotation_cogmap(item) -> Tuple[str, list, list]:
    """
    Generate the cogmap for rotation setting.
    """
    id = item.get("id", "")
    category = item.get("category", [])
    type = item.get("type", "")
    objects = item.get("meta_info", [])
    question = item.get("question", "")
    images = item.get("images", [])
    gt_answer = item.get("answer", "")
    # we have these configurations: {'three_view': 345, 'two_view_clockwise': 220, 'two_view_counterclockwise': 120, 'four_view': 360, 'two_view_opposite': 36}
    objects_coordinates = []
    view_coordinates = []

    view_base_name = "Image" if "image" in question.lower() else "View"

    if "two" in type:
        assert len(objects) == 2, f"Two objects are expected for two view rotation, but got {len(objects)}"
        if type == 'two_view_clockwise': # example, front to right
            objects_coordinates = [
                gen_object_coordinate_dict(objects[0], [5, 3]),
                gen_object_coordinate_dict(objects[1], [7, 5])
            ]
            view_coordinates = [
                {"name": f"{view_base_name} 1", "position": [5, 5], "facing": "up"},
                {"name": f"{view_base_name} 2", "position": [5, 5], "facing": "right"}
            ]
        elif type == 'two_view_counterclockwise': # example, front to left
            objects_coordinates = [
                gen_object_coordinate_dict(objects[0], [5, 3]),
                gen_object_coordinate_dict(objects[1], [3, 5])
            ]
            view_coordinates = [
                {"name": f"{view_base_name} 1", "position": [5, 5], "facing": "up"},
                {"name": f"{view_base_name} 2", "position": [5, 5], "facing": "left"}
            ]
        elif type == 'two_view_opposite': # opposite means 180 degree rotation
            objects_coordinates = [
                gen_object_coordinate_dict(objects[0], [5, 3]),
                gen_object_coordinate_dict(objects[1], [5, 7])
            ]
            view_coordinates = [
                {"name": f"{view_base_name} 1", "position": [5, 5], "facing": "up"},
                {"name": f"{view_base_name} 2", "position": [5, 5], "facing": "down"}
            ]
    elif "three" in type:
        assert len(objects) == 3, f"Three objects are expected for three view rotation, but got {len(objects)}"
        assert type == 'three_view', f"Unknown type: {type}" # clockwise, example, front to right to back
        objects_coordinates = [
            gen_object_coordinate_dict(objects[0], [3, 5]),
            gen_object_coordinate_dict(objects[1], [5, 3]),
            gen_object_coordinate_dict(objects[2], [7, 5])
        ]
        view_coordinates = [
            {"name": f"{view_base_name} 1", "position": [5, 5], "facing": "left"},
            {"name": f"{view_base_name} 2", "position": [5, 5], "facing": "up"},
            {"name": f"{view_base_name} 3", "position": [5, 5], "facing": "right"}
        ]
    elif "four" in type:
        assert len(objects) == 4, f"Four objects are expected for four view rotation, but got {len(objects)}"
        assert type == 'four_view', f"Unknown type: {type}" # example, front to right to back to left
        objects_coordinates = [
            gen_object_coordinate_dict(objects[0], [3, 5]),
            gen_object_coordinate_dict(objects[1], [5, 3]),
            gen_object_coordinate_dict(objects[2], [7, 5]),
            gen_object_coordinate_dict(objects[3], [5, 7])
        ]
        view_coordinates = [
            {"name": f"{view_base_name} 1", "position": [5, 5], "facing": "left"},
            {"name": f"{view_base_name} 2", "position": [5, 5], "facing": "up"},
            {"name": f"{view_base_name} 3", "position": [5, 5], "facing": "right"},
            {"name": f"{view_base_name} 4", "position": [5, 5], "facing": "down"}
        ]
    else:
        raise ValueError(f"Unknown type: {type}")
    
    cogmap = {
        "objects": objects_coordinates,
        "views": view_coordinates
    }
    
    return format_cogmap_json(cogmap), objects, []


def generate_among_cogmap_with_options(item, suppress_warnings: bool = False) -> Tuple[str, list, list]:
    """
    Wrapper for generate_among_cogmap that supports suppress_warnings option.
    """
    # Temporarily store the original print function
    original_print = print
    
    # If suppress_warnings is True, replace print with a no-op for warnings
    if suppress_warnings:
        def conditional_print(*args, **kwargs):
            message = ' '.join(str(arg) for arg in args)
            if not message.startswith("Warning:"):
                original_print(*args, **kwargs)
        
        # Replace print globally for this function call
        import builtins
        builtins.print = conditional_print
    
    try:
        result = generate_among_cogmap(item)
    finally:
        # Restore original print function
        if suppress_warnings:
            import builtins
            builtins.print = original_print
    
    return result


class CogMapGenerator:
    """
    Main cognitive map generator class.
    
    Provides a clean interface to the original cognitive map generation logic
    while preserving all original functionality.
    """
    
    def __init__(self, format_type: str = "full", suppress_warnings: bool = False):
        """
        Initialize generator.
        
        Args:
            format_type: "full" or "shortened" cognitive map format
            suppress_warnings: Whether to suppress warning messages
        """
        self.format_type = format_type
        self.suppress_warnings = suppress_warnings
        
        # Map setting names to generation functions
        self.generators = {
            "around": generate_around_cogmap,
            "among": generate_among_cogmap,
            "translation": generate_translation_cogmap,
            "rotation": generate_rotation_cogmap
        }
    
    def detect_setting(self, item_id: str) -> Optional[str]:
        """
        Detect spatial reasoning setting from item ID.
        
        Args:
            item_id: Item identifier
            
        Returns:
            Setting name or None if not detected
        """
        for setting in self.generators.keys():
            if setting in item_id:
                return setting
        return None
    
    def generate_for_item(self, item: Dict) -> Tuple[str, List[str], List[str]]:
        """
        Generate cognitive map for a single item.
        
        Args:
            item: Data item dictionary
            
        Returns:
            Tuple of (cogmap_string, main_objects, oriented_objects)
        """
        item_id = item.get("id", "")
        setting = self.detect_setting(item_id)
        
        if not setting:
            raise ValueError(f"Could not detect setting for item ID: {item_id}")
        
        generator_func = self.generators[setting]
        
        # For among setting, pass suppress_warnings parameter
        if setting == "among":
            return generate_among_cogmap_with_options(item, self.suppress_warnings)
        else:
            return generator_func(item)
    
    def add_cogmap_to_item(self, item: Dict) -> Dict:
        """
        Add cognitive map fields to an item.
        
        Args:
            item: Original data item
            
        Returns:
            Item with cognitive map fields added
        """
        try:
            cogmap_str, main_objects, oriented_objects = self.generate_for_item(item)
            
            # Determine format-specific configurations
            item_id = item.get("id", "")
            setting = self.detect_setting(item_id)
            
            birdview_input_str = "" if setting == "translation" else "\n- The map is shown in the bird's view"
            birdview_output_str = "" if setting == "translation" else " in the bird's view"
            
            orientation_info = f"4. For objects [{' '.join(oriented_objects)}], determine their facing direction as up, right, down, or left. For other objects, omit the facing direction." if len(oriented_objects) > 0 else ""
            
            # Add cognitive map fields with new naming scheme
            item["grounded_cogmap"] = cogmap_str
            item["grounded_cogmap_description"] = COG_MAP_DESCRIPTION_FOR_INPUT.replace("{birdview}", birdview_input_str)
            
            # Generate both augmented and plain instruction prompts
            aug_cogmap_instruction = COG_MAP_DESCRIPTION_FOR_OUTPUT.replace("{birdview}", birdview_output_str)
            aug_cogmap_instruction = aug_cogmap_instruction.replace("<orientation_info>", orientation_info)
            aug_cogmap_instruction = aug_cogmap_instruction.replace("{obj}", ", ".join(main_objects))
            
            plain_cogmap_instruction = COG_MAP_DESCRIPTION_FOR_OUTPUT_SHORTEN.replace("{birdview}", birdview_output_str)
            plain_cogmap_instruction = plain_cogmap_instruction.replace("<orientation_info>", orientation_info)
            plain_cogmap_instruction = plain_cogmap_instruction.replace("{obj}", ", ".join(main_objects))
            
            item["aug_cogmap_gen_instruction"] = aug_cogmap_instruction
            item["plain_cogmap_gen_instruction"] = plain_cogmap_instruction
            
            return item
            
        except Exception as e:
            print(f"Error generating cognitive map for item {item.get('id', 'unknown')}: {e}")
            return item 


class ReasoningChainGenerator:
    """
    Reasoning Chain Generator - TO BE IMPLEMENTED
    
    This class will handle the generation of synthetic reasoning chains
    for different spatial reasoning settings. The original reasoning 
    generation scripts for each setting will be integrated here.
    
    Settings to support:
    - Around: reasoning for object positions relative to viewpoints
    - Among: reasoning for objects distributed among viewpoints  
    - Translation: reasoning for object movement scenarios
    - Rotation: reasoning for object rotation scenarios
    """
    
    def __init__(self, setting: str):
        """
        Initialize reasoning generator for specific setting.
        
        Args:
            setting: Spatial reasoning setting ("around", "among", "translation", "rotation")
        """
        self.setting = setting
        # TODO: Add setting-specific configuration
        
    def generate_reasoning_chain(self, item: Dict) -> str:
        """
        Generate synthetic reasoning chain for an item.
        
        Args:
            item: Data item with question and spatial information
            
        Returns:
            Generated reasoning chain text
            
        Note:
            This method is a placeholder. The actual implementation will
            be added when the original reasoning scripts are integrated.
        """
        # TODO: Implement reasoning generation logic for each setting
        # This will integrate the original reasoning generation scripts
        
        setting_functions = {
            "around": self._generate_around_reasoning,
            "among": self._generate_among_reasoning,
            "translation": self._generate_translation_reasoning,
            "rotation": self._generate_rotation_reasoning
        }
        
        if self.setting in setting_functions:
            return setting_functions[self.setting](item)
        else:
            raise ValueError(f"Unsupported reasoning setting: {self.setting}")
    
    def _generate_around_reasoning(self, item: Dict) -> str:
        """Generate reasoning chain for 'around' setting - TO BE IMPLEMENTED"""
        # TODO: Integrate original around reasoning generation script
        return "[PLACEHOLDER] Around reasoning chain will be generated here"
    
    def _generate_among_reasoning(self, item: Dict) -> str:
        """Generate reasoning chain for 'among' setting - TO BE IMPLEMENTED"""
        # TODO: Integrate original among reasoning generation script  
        return "[PLACEHOLDER] Among reasoning chain will be generated here"
    
    def _generate_translation_reasoning(self, item: Dict) -> str:
        """Generate reasoning chain for 'translation' setting - TO BE IMPLEMENTED"""
        # TODO: Integrate original translation reasoning generation script
        return "[PLACEHOLDER] Translation reasoning chain will be generated here"
    
    def _generate_rotation_reasoning(self, item: Dict) -> str:
        """Generate reasoning chain for 'rotation' setting - TO BE IMPLEMENTED"""
        # TODO: Integrate original rotation reasoning generation script
        return "[PLACEHOLDER] Rotation reasoning chain will be generated here"
    
    def add_reasoning_to_item(self, item: Dict) -> Dict:
        """
        Add reasoning chain field to an item.
        
        Args:
            item: Original data item
            
        Returns:
            Item with reasoning field added
        """
        try:
            reasoning_chain = self.generate_reasoning_chain(item)
            item["reasoning_chain"] = reasoning_chain
            return item
        except Exception as e:
            print(f"Error generating reasoning chain for item {item.get('id', 'unknown')}: {e}")
            return item


def generate_with_reasoning(item: Dict, cogmap_format: str = "full") -> Dict:
    """
    Generate both cognitive map and reasoning chain for an item.
    
    Args:
        item: Data item
        cogmap_format: Cognitive map format type ("full", "shortened", "qwen")
        
    Returns:
        Item with both cognitive map and reasoning chain added
        
    Note:
        Reasoning generation is currently placeholder. Will be implemented
        when original reasoning scripts are integrated.
    """
    # Type validation for cogmap_format
    valid_formats = ["full", "shortened", "qwen"]
    if cogmap_format not in valid_formats:
        print(f"Warning: Invalid format '{cogmap_format}', using 'full' instead")
        cogmap_format = "full"
    
    # Generate cognitive map
    cogmap_generator = CogMapGenerator(format_type=cogmap_format)
    item_with_cogmap = cogmap_generator.add_cogmap_to_item(item.copy())
    
    # Generate reasoning chain (placeholder implementation)
    item_id = item.get("id", "")
    setting = cogmap_generator.detect_setting(item_id)
    
    if setting:
        reasoning_generator = ReasoningChainGenerator(setting)
        item_with_reasoning = reasoning_generator.add_reasoning_to_item(item_with_cogmap)
        return item_with_reasoning
    else:
        print(f"Warning: Could not detect setting for reasoning generation: {item_id}")
        return item_with_cogmap 