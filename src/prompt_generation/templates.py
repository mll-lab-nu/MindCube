"""
Prompt Templates

Defines various prompt templates and formatting utilities for different task settings.
Based on the original prompt structures but modularized for flexibility.
"""

from typing import Dict, List, Optional
import json

# Question section header
QUESTION_HEADER = "[Question]\n"

RAW_QA_BACKGROUND_INSTRUCTION = """[Task]
Your task is to analyze the spatial arrangement of objects in the scene by examining the provided images, which show the scene from different viewpoints.
[Answer Instruction]
You only need to provide *ONE* correct answer selecting from the options listed below. For example, if you think the correct answer is 'A. Above' from 'A. Above B. Under C. Front D. Behind', your response should **only** be '<answer>A. Above</answer>'.
"""

FF_RSN_BACKGROUND_INSTRUCTION = '''[Task]
Your task is to analyze the spatial arrangement of objects in the scene by examining the provided images, which show the scene from different viewpoints.
[Answer Instruction]
Please do step by step reasoning first, then give your final answer. For example, if you think the correct answer is 'A. Above' from 'A. Above B. Under C. Front D. Behind', your response should be this format: '<think>(replace with your reasoning here)</think><answer>A. Above</answer>'.
'''

AUG_CGMAP_IN_BACKGROUND_INSTRUCTION = '''[Task]
Your task is to analyze the spatial arrangement of objects in the scene by examining the provided images, which show the scene from different viewpoints. Also, we provide you a cognitive map that shows the general layout for the scene. Please use the cognitive map to reason and answer the question.
[Answer Instruction]
You only need to provide *ONE* correct answer selecting from the options listed below. For example, if you think the correct answer is 'A. Above' from 'A. Above B. Under C. Front D. Behind', your response should **only** be '<answer>A. Above</answer>'.
<cogmap_description>
Below is the cognitive map of the scene related to the question. Please use it to reason and answer the question.
<grounded_cogmap>
'''

AUG_CGMAP_OUT_BACKGROUND_INSTRUCTION = '''<replace_here>
[Answer Instruction]
1. Given the provided views and main objects mentioned in the above rules, you **MUST** present your cognitive map in the following JSON format **before your answer**:
```json
{
  "objects": [
    {"name": "object_name", "position": [x, y], "facing": "direction"},
    {"name": "object_without_orientation", "position": [x, y]}
  ],
  "views": [
    {"name": "View 1", "position": [x, y], "facing": "direction"},
    {"name": "View 2", "position": [x, y], "facing": "direction"}
  ]
}
```
2. Next, provide *ONE* correct answer selecting from the options. Your answer field must be in the format like "A. Above"
3. In general, your response's format should be like "Based on my observation, the answer is:\n<cogmap>(Replace with your cogmap here)</cogmap><answer>(Replace with your answer here)</answer>". Your option must be from the available options.
'''

PLAIN_CGMAP_OUT_BACKGROUND_INSTRUCTION = '''<replace_here>
[Answer Instruction]
1. Given the provided views and main objects mentioned in the above rules, you **MUST** present your cognitive map in the following JSON format **before your answer**:
```json
{
    "object_category_1": {"position": [x, y]},
    "object_category_2": {"position": [x, y], "facing": "direction"}, # if the object is asked for orientation
    ...
}
```
2. Next, provide *ONE* correct answer selecting from the options. Your answer field must be in the format like "A. Above"
3. In general, your response's format should be like "Based on my observation, the answer is:\n<cogmap>(Replace with your cogmap here)</cogmap><answer>(Replace with your answer here)</answer>". Your option must be from the available options.'''

PLAIN_CGMAP_FFR_OUT_BACKGROUND_INSTRUCTION = '''<replace_here>
[Answer Instruction]
1. Given the provided views and main objects mentioned in the above rules, you **MUST** present your cognitive map in the following JSON format **before your reasoning**:
```json
{
    "object_category_1": {"position": [x, y]},
    "object_category_2": {"position": [x, y], "facing": "direction"}, # if the object is asked for orientation
    ...
}
```
2. Next, please also provide your reasons step by step in details, then provide *ONE* correct answer selecting from the options. Your answer field must be in the format like "A. Above"
3. In general, your response's format should be like "Based on my observation, the answer is:\n<cogmap>(Replace with your cogmap here)</cogmap><think>(Replace with your reasoning here)</think><answer>(Replace with your answer here)</answer>". Your option must be from the available options.'''

AUG_CGMAP_FFR_OUT_BACKGROUND_INSTRUCTION = '''<replace_here>
[Answer Instruction]
1. Given the provided views and main objects mentioned in the above rules, you **MUST** present your cognitive map in the following JSON format **before your reasoning**:
```json
{
  "objects": [
    {"name": "object_name", "position": [x, y], "facing": "direction"},
    {"name": "object_without_orientation", "position": [x, y]}
  ],
  "views": [
    {"name": "View 1", "position": [x, y], "facing": "direction"},
    {"name": "View 2", "position": [x, y], "facing": "direction"}
  ]
}
```
2. Next, please also provide your reasons step by step in details, then provide *ONE* correct answer selecting from the options. Your answer field must be in the format like "A. Above"
3. In general, your response's format should be like "Based on my observation, the answer is:\n<cogmap>(Replace with your cogmap here)</cogmap><think>(Replace with your reasoning here)</think><answer>(Replace with your answer here)</answer>". Your option must be from the available options.'''

CGMAP_IN_FFR_OUT_BACKGROUND_INSTRUCTION = '''[Task]
Your task is to analyze the spatial arrangement of objects in the scene by examining the provided images, which show the scene from different viewpoints. Also, we provide you a cognitive map that shows the general layout for the scene. Please use the cognitive map to reason and answer the question.
[Answer Instruction]
Please do step by step reasoning first, then give your final answer. For example, if you think the correct answer is 'A. Above' from 'A. Above B. Under C. Front D. Behind', your response should be this format: '<think>(replace with your reasoning here)</think><answer>A. Above</answer>'.
<cogmap_description>
Below is the cognitive map of the scene related to the question. Please use it to reason and answer the question.
<grounded_cogmap>'''

CGMAP_IN_CGMAP_OUT_BACKGROUND_INSTRUCTION = '''[Task]
Your task is to analyze the spatial arrangement of objects in the scene by examining the provided images, which show the scene from different viewpoints. Also, we provide you a cognitive map that shows the general layout for the scene. Please use the cognitive map to reason and answer the question.
[Answer Instruction]
1. Based on the provided cognitive map and your analysis, you **MUST** present your updated/refined cognitive map in the following JSON format **before your answer**:
```json
{
  "objects": [
    {"name": "object_name", "position": [x, y], "facing": "direction"},
    {"name": "object_without_orientation", "position": [x, y]}
  ],
  "views": [
    {"name": "View 1", "position": [x, y], "facing": "direction"},
    {"name": "View 2", "position": [x, y], "facing": "direction"}
  ]
}
```
2. Next, provide *ONE* correct answer selecting from the options. Your answer field must be in the format like "A. Above"
3. In general, your response's format should be like "Based on my observation, the answer is:\n<cogmap>(Replace with your cogmap here)</cogmap><answer>(Replace with your answer here)</answer>". Your option must be from the available options.
<cogmap_description>
Below is the cognitive map of the scene related to the question. Please use it to reason and answer the question.
<grounded_cogmap>'''

# NL PROMPT VARIATIONS

NOFORMAT_FF_RSN_BACKGROUND_INSTRUCTION = '''[Task]
Your task is to analyze the spatial arrangement of objects in the scene by examining the provided images, which show the scene from different viewpoints.
[Answer Instruction]
Please do step by step reasoning first, then give your final answer. For example, if you think the correct answer is 'A. Above' from 'A. Above B. Under C. Front D. Behind', your response should be this format: '(replace with your reasoning here) And my answer is: A. Above'.
'''


NL_CGMAP_IN_FFR_OUT_BACKGROUND_INSTRUCTION = '''[Task]
Your task is to analyze the spatial arrangement of objects in the scene by examining the provided images, which show the scene from different viewpoints. Also, we provide you a cognitive map that shows the general layout for the scene. Please use the cognitive map to reason and answer the question.
[Answer Instruction]
Please do step by step reasoning first, then give your final answer. For example, if you think the correct answer is 'A. Above' from 'A. Above B. Under C. Front D. Behind', your response should be this format: '<think>(replace with your reasoning here)</think><answer>A. Above</answer>'.
<cogmap_description>
Below is the natural language description of the cognitive map of the scene related to the question. Please use it to reason and answer the question.
<grounded_cogmap>'''

NL_GROUNDED_CGMAP_DESCRIPTION = '''[Cognitive Map Format]
We provide you a natural language description of the cognitive map for the scene that is related to the question you should answer. Below is the description of the spatial layout:

- The scene is described using a 10x10 grid coordinate system where [0,0] is at the top-left corner and [9,9] is at the bottom-right corner
- The layout is shown from a bird's eye view perspective
- Spatial directions are defined as:
  * up = towards the top of the grid (decreasing y-value)
  * right = towards the right of the grid (increasing x-value)
  * down = towards the bottom of the grid (increasing y-value)
  * left = towards the left of the grid (decreasing x-value)
  * inner = straight into the 2D map (perpendicular to the grid, pointing away from you)
  * outer = straight out of the 2D map (perpendicular to the grid, pointing towards you)
- Object descriptions include their positions and orientations within the scene
- Camera viewpoint descriptions indicate where each view was captured from'''

NL_AUG_CGMAP_GEN_INSTRUCTION = '''[Task]
Your task is to analyze the spatial arrangement of objects in the scene by examining the provided images, which show the scene from different viewpoints. You will then create a detailed cognitive map representing the scene using a 10x10 grid coordinate system.

[Rules]
1. Focus ONLY on these categories of objects in the scene: {black sneaker, light purple sofa, brown curtains and windows, TV, wooden dining table}
2. Create a cognitive map with the following structure in the bird's view:
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
4. For objects [black sneaker light purple sofa], determine their facing direction as up, right, down, or left. For other objects, omit the facing direction.'''

NL_PLAIN_CGMAP_GEN_INSTRUCTION = '''[Task]
Your task is to analyze the spatial arrangement of objects in the scene by examining the provided images, which show the scene from different viewpoints. You will then create a detailed cognitive map representing the scene using a **10x10 grid coordinate system**.

[Rules]
1. Focus ONLY on these categories of objects in the scene: {black sneaker, light purple sofa, brown curtains and windows, TV, wooden dining table}
2. Create a cognitive map with the following structure in the bird's view:
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
4. For objects [black sneaker light purple sofa], determine their facing direction as up, right, down, or left. For other objects, omit the facing direction.'''

NL_AUG_CGMAP_FFR_OUT_BACKGROUND_INSTRUCTION = '''<replace_here>
[Answer Instruction]
1. Given the provided views and main objects mentioned in the above rules, you **MUST** present your cognitive map in **natural language description format** **before your reasoning**:
   - Describe the positions of all objects using the 10x10 grid coordinate system
   - Include camera viewpoint descriptions where applicable
   - Maintain spatial accuracy and relative positioning
2. Next, please also provide your reasons step by step in details, then provide *ONE* correct answer selecting from the options. Your answer field must be in the format like "A. Above"
3. In general, your response's format should be like "Based on my observation, the answer is:\n<cogmap>(Replace with your natural language cognitive map description here)</cogmap><think>(Replace with your reasoning here)</think><answer>(Replace with your answer here)</answer>". Your option must be from the available options.'''

NL_PLAIN_CGMAP_FFR_OUT_BACKGROUND_INSTRUCTION = '''<replace_here>
[Answer Instruction]
1. Given the provided views and main objects mentioned in the above rules, you **MUST** present your cognitive map in **natural language description format** **before your reasoning**:
   - Describe the positions of all objects using the 10x10 grid coordinate system
   - Maintain spatial accuracy and relative positioning
2. Next, please also provide your reasons step by step in details, then provide *ONE* correct answer selecting from the options. Your answer field must be in the format like "A. Above"
3. In general, your response's format should be like "Based on my observation, the answer is:\n<cogmap>(Replace with your natural language cognitive map description here)</cogmap><think>(Replace with your reasoning here)</think><answer>(Replace with your answer here)</answer>". Your option must be from the available options.'''


def convert_aug_cogmap_to_nl_cogmap(aug_cogmap: str) -> str:
    '''
    Convert the augmented cogmap to a natural language description.
    The augmented cogmap is in the following format:
    {
        "objects": [
            {"name": "object_name", "position": [x, y], "facing": "direction"},
            {"name": "object_without_orientation", "position": [x, y]}
        ],
        "views": [
            {"name": "View 1", "position": [x, y], "facing": "direction"},
            {"name": "View 2", "position": [x, y], "facing": "direction"}
        ]
    }
    '''
    try:
        # Parse the JSON string
        cogmap_data = json.loads(aug_cogmap)
        
        description_parts = []
        
        # Describe objects
        if "objects" in cogmap_data and cogmap_data["objects"]:
            description_parts.append("Objects in the scene:")
            for obj in cogmap_data["objects"]:
                name = obj.get("name", "unknown object")
                position = obj.get("position", [0, 0])
                facing = obj.get("facing")
                
                if facing:
                    obj_desc = f"- {name} is located at position ({position[0]}, {position[1]}) and is facing {facing}"
                else:
                    obj_desc = f"- {name} is located at position ({position[0]}, {position[1]})"
                description_parts.append(obj_desc)
        
        # Describe views
        if "views" in cogmap_data and cogmap_data["views"]:
            description_parts.append("\nCamera views:")
            for view in cogmap_data["views"]:
                view_name = view.get("name", "unknown view")
                position = view.get("position", [0, 0])
                facing = view.get("facing", "unknown direction")
                
                view_desc = f"- {view_name} is positioned at ({position[0]}, {position[1]}) looking towards {facing}"
                description_parts.append(view_desc)
        
        return "\n".join(description_parts)
        
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return f"Error parsing augmented cognitive map: {str(e)}"
    
def convert_plain_cogmap_to_nl_cogmap(plain_cogmap: str) -> str:
    '''
    Convert the plain cogmap to a natural language description.
    The plain cogmap is in the following format:
    {
        "object_category_1": {"position": [x, y]},
        "object_category_2": {"position": [x, y], "facing": "direction"}, # if the object is asked for orientation
        ...
    }
    '''
    try:
        # Parse the JSON string
        cogmap_data = json.loads(plain_cogmap)
        
        description_parts = ["Objects in the scene:"]
        
        # Describe each object category
        for obj_name, obj_info in cogmap_data.items():
            position = obj_info.get("position", [0, 0])
            facing = obj_info.get("facing")
            
            if facing:
                obj_desc = f"- {obj_name} is located at position ({position[0]}, {position[1]}) and is facing {facing}"
            else:
                obj_desc = f"- {obj_name} is located at position ({position[0]}, {position[1]})"
            description_parts.append(obj_desc)
        
        return "\n".join(description_parts)
        
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return f"Error parsing plain cognitive map: {str(e)}"


class PromptTemplate:
    """Base class for prompt templates."""
    
    def __init__(self, name: str):
        self.name = name
    
    def format_question(self, question: str) -> str:
        """Format the question part of the prompt."""
        # Process question similar to original logic
        formatted = question
        
        return QUESTION_HEADER + formatted
    
    def generate_prompt(self, data: Dict) -> str:
        """Generate the complete prompt. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement generate_prompt")
    
    def generate_output(self, data: Dict) -> str:
        """Generate the grounded output. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement generate_output")
    
    def _extract_answer_text(self, question: str, answer_key: str) -> str:
        """Extract answer text from question options."""
        import re
        pattern = r"([A-Z]\.\s+.*?)(?=[A-Z]\.|$)"
        matches = re.findall(pattern, question)
        for match in matches:
            if match.strip().startswith(answer_key + '.'):
                return match.strip()
        raise ValueError(f"Answer key {answer_key} not found in question")

class RawQATemplate(PromptTemplate):
    """Template for raw QA without cognitive maps or reasoning chains."""
    
    def __init__(self):
        super().__init__("raw_qa")
    
    def generate_prompt(self, data: Dict) -> str:
        """Generate raw QA prompt."""
        question = data.get("question", "")
        
        prompt_parts = [
            RAW_QA_BACKGROUND_INSTRUCTION,
            self.format_question(question)
        ]
        
        return "\n".join(prompt_parts)
    
    def generate_output(self, data: Dict) -> str:
        """Generate target output for raw QA."""
        gt_answer = data.get("gt_answer", "")
        
        # Extract answer text from options if available
        question = data.get("question", "")
        answer = self._extract_answer_text(question, gt_answer)
        
        return f"<answer>{answer}</answer>"
    
class FFRSNTemplate(PromptTemplate):
    """Template for FF-RSN (free form reasoning)."""
    def __init__(self):
        super().__init__("ff_rsn")
    
    def generate_prompt(self, data: Dict) -> str:
        """Generate FF-RSN prompt."""
        question = data.get("question", "")
        
        prompt_parts = [
            FF_RSN_BACKGROUND_INSTRUCTION,
            self.format_question(question)
        ]
        
        return "\n".join(prompt_parts)
    
    def generate_output(self, data: Dict) -> str:
        """Generate target output for FF-RSN."""
        gt_answer = data.get("gt_answer", "")
        question = data.get("question", "")
        reasoning_chain = data.get("reasoning_chain", "")
        answer = self._extract_answer_text(question, gt_answer)
        
        return f"<think>{reasoning_chain}</think><answer>{answer}</answer>"
    

class AugCGMapInTemplate(PromptTemplate):
    """Template for Aug-CGMap-In tasks. In this task, we input the question with the augmented cognitive map."""
    def __init__(self):
        super().__init__("aug_cgmap_in")
    
    def generate_prompt(self, data: Dict) -> str:
        """Generate Aug-CGMap-In prompt."""
        question = data.get("question", "")
        cogmap_description = data.get("grounded_cogmap_description", "")
        grounded_cogmap = data.get("grounded_cogmap", "")
        
        prompt_parts = [
            AUG_CGMAP_IN_BACKGROUND_INSTRUCTION.replace("<cogmap_description>", cogmap_description).replace("<grounded_cogmap>", grounded_cogmap),
            self.format_question(question)
        ]
        
        return "\n".join(prompt_parts)
    
    def generate_output(self, data: Dict) -> str:
        """Generate target output for Aug-CGMap-In."""
        gt_answer = data.get("gt_answer", "")
        question = data.get("question", "")
        answer = self._extract_answer_text(question, gt_answer)
        
        return f"<answer>{answer}</answer>"

class AugCGMapOutTemplate(PromptTemplate):
    """Template for Aug-CGMap-Out tasks."""
    def __init__(self):
        super().__init__("aug_cgmap_out")
    
    def generate_prompt(self, data: Dict) -> str:
        """Generate Aug-CGMap-Out prompt."""
        question = data.get("question", "")
        aug_cogmap_gen_instruction = data.get("aug_cogmap_gen_instruction", "")
        
        prompt_parts = [
            AUG_CGMAP_OUT_BACKGROUND_INSTRUCTION.replace("<replace_here>", aug_cogmap_gen_instruction),
            self.format_question(question)
        ]
        
        return "\n".join(prompt_parts)
    
    def generate_output(self, data: Dict) -> str:
        """Generate target output for Aug-CGMap-Out."""
        gt_answer = data.get("gt_answer", "")
        grounded_cogmap = data.get("grounded_cogmap", "")
        question = data.get("question", "")
        answer = self._extract_answer_text(question, gt_answer)
        
        return f"Based on my observation, the answer is:\n<cogmap>```json\n{grounded_cogmap}\n```</cogmap><answer>{answer}</answer>"

class PlainCGMapOutTemplate(PromptTemplate):
    """Template for Plain-CGMap-Out tasks."""
    def __init__(self):
        super().__init__("plain_cgmap_out")
    
    def generate_prompt(self, data: Dict) -> str:
        """Generate Plain-CGMap-Out prompt."""
        question = data.get("question", "")
        plain_cogmap_gen_instruction = data.get("plain_cogmap_gen_instruction", "")
        
        prompt_parts = [
            PLAIN_CGMAP_OUT_BACKGROUND_INSTRUCTION.replace("<replace_here>", plain_cogmap_gen_instruction),
            self.format_question(question)
        ]
        
        return "\n".join(prompt_parts)
    
    def generate_output(self, data: Dict) -> str:
        """Generate target output for Plain-CGMap-Out."""
        gt_answer = data.get("gt_answer", "")
        question = data.get("question", "")
        aug_grounded_cogmap = data.get("grounded_cogmap", "")
        answer = self._extract_answer_text(question, gt_answer)
        
        # Convert augmented cogmap to plain cogmap
        plain_cogmap = self._convert_to_plain_cogmap(aug_grounded_cogmap)
        
        return f"Based on my observation, the answer is:\n<cogmap>```json\n{plain_cogmap}\n```</cogmap><answer>{answer}</answer>"
    
    def _convert_to_plain_cogmap(self, grounded_cogmap: str) -> str:
        """Convert the grounded augmented cogmap to a plain cogmap."""
        try:
            # Parse the augmented cogmap JSON
            aug_data = json.loads(grounded_cogmap)
            
            # Manually build the plain cogmap JSON string
            json_lines = ["{"]
            
            # Convert objects array to plain format
            if "objects" in aug_data:
                object_entries = []
                for obj in aug_data["objects"]:
                    obj_name = obj.get("name", "")
                    position = obj.get("position", [0, 0])
                    
                    # Build position string
                    pos_str = f"[{position[0]}, {position[1]}]"
                    
                    # Build object entry
                    if "facing" in obj:
                        facing = obj["facing"]
                        entry = f'    "{obj_name}": {{"position": {pos_str}, "facing": "{facing}"}}'
                    else:
                        entry = f'    "{obj_name}": {{"position": {pos_str}}}'
                    
                    object_entries.append(entry)
                
                # Join entries with commas
                for i, entry in enumerate(object_entries):
                    if i < len(object_entries) - 1:
                        json_lines.append(entry + ",")
                    else:
                        json_lines.append(entry)
            
            json_lines.append("}")
            
            return "\n".join(json_lines)
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Return empty object if parsing fails
            print(f"Error converting to plain cogmap: {e}")
            exit(1)

class PlainCGMapFFROutTemplate(PromptTemplate):
    """Template for Plain-CGMap-FFR-Out tasks."""
    def __init__(self):
        super().__init__("plain_cgmap_ffr_out")
    
    def generate_prompt(self, data: Dict) -> str:
        """Generate Plain-CGMap-FFR-Out prompt."""
        question = data.get("question", "")
        plain_cogmap_gen_instruction = data.get("plain_cogmap_gen_instruction", "")
        
        prompt_parts = [
            PLAIN_CGMAP_FFR_OUT_BACKGROUND_INSTRUCTION.replace("<replace_here>", plain_cogmap_gen_instruction),
            self.format_question(question)
        ]
        
        return "\n".join(prompt_parts)
    
    def generate_output(self, data: Dict) -> str:
        """Generate target output for Plain-CGMap-FFR-Out."""
        gt_answer = data.get("gt_answer", "")
        question = data.get("question", "")
        reasoning_chain = data.get("reasoning_chain", "")
        aug_grounded_cogmap = data.get("grounded_cogmap", "")
        # Convert augmented cogmap to plain cogmap
        plain_cogmap = self._convert_to_plain_cogmap(aug_grounded_cogmap)
        
        answer = self._extract_answer_text(question, gt_answer)
        
        return f"Based on my observation, the answer is:\n<cogmap>```json\n{plain_cogmap}\n```</cogmap><think>{reasoning_chain}</think><answer>{answer}</answer>"
    
    def _convert_to_plain_cogmap(self, grounded_cogmap: str) -> str:
        """Convert the grounded augmented cogmap to a plain cogmap."""
        try:
            # Parse the augmented cogmap JSON
            aug_data = json.loads(grounded_cogmap)
            
            # Manually build the plain cogmap JSON string
            json_lines = ["{"]
            
            # Convert objects array to plain format
            if "objects" in aug_data:
                object_entries = []
                for obj in aug_data["objects"]:
                    obj_name = obj.get("name", "")
                    position = obj.get("position", [0, 0])
                    
                    # Build position string
                    pos_str = f"[{position[0]}, {position[1]}]"
                    
                    # Build object entry
                    if "facing" in obj:
                        facing = obj["facing"]
                        entry = f'    "{obj_name}": {{"position": {pos_str}, "facing": "{facing}"}}'
                    else:
                        entry = f'    "{obj_name}": {{"position": {pos_str}}}'
                    
                    object_entries.append(entry)
                
                # Join entries with commas
                for i, entry in enumerate(object_entries):
                    if i < len(object_entries) - 1:
                        json_lines.append(entry + ",")
                    else:
                        json_lines.append(entry)
            
            json_lines.append("}")
            
            return "\n".join(json_lines)
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Return empty object if parsing fails
            print(f"Error converting to plain cogmap: {e}")
            exit(1)
        
class AugCGMapFFROutTemplate(PromptTemplate):
    """Template for Aug-CGMap-FFR-Out tasks."""
    def __init__(self):
        super().__init__("aug_cgmap_ffr_out")
    
    def generate_prompt(self, data: Dict) -> str:
        """Generate Aug-CGMap-FFR-Out prompt."""
        question = data.get("question", "")
        aug_cogmap_gen_instruction = data.get("aug_cogmap_gen_instruction", "")
        
        prompt_parts = [
            AUG_CGMAP_FFR_OUT_BACKGROUND_INSTRUCTION.replace("<replace_here>", aug_cogmap_gen_instruction),
            self.format_question(question)
        ]
        
        return "\n".join(prompt_parts)
    
    def generate_output(self, data: Dict) -> str:
        """Generate target output for Aug-CGMap-FFR-Out."""
        gt_answer = data.get("gt_answer", "")
        question = data.get("question", "")
        reasoning_chain = data.get("reasoning_chain", "")
        grounded_cogmap = data.get("grounded_cogmap", "")
        
        answer = self._extract_answer_text(question, gt_answer)
        
        return f"Based on my observation, the answer is:\n<cogmap>```json\n{grounded_cogmap}\n```</cogmap><think>{reasoning_chain}</think><answer>{answer}</answer>"


class CGMapInFFROutTemplate(PromptTemplate):
    """Template for CGMap-In-FFR-Out tasks."""
    def __init__(self):
        super().__init__("cgmap_in_ffr_out")
    
    def generate_prompt(self, data: Dict) -> str:
        """Generate CGMap-In-FFR-Out prompt."""
        question = data.get("question", "")
        cogmap_description = data.get("grounded_cogmap_description", "")
        grounded_cogmap = data.get("grounded_cogmap", "")
        
        prompt_parts = [
            CGMAP_IN_FFR_OUT_BACKGROUND_INSTRUCTION.replace("<cogmap_description>", cogmap_description).replace("<grounded_cogmap>", grounded_cogmap),
            self.format_question(question)
        ]
        
        return "\n".join(prompt_parts)
    
    def generate_output(self, data: Dict) -> str:
        """Generate target output for CGMap-In-FFR-Out."""
        gt_answer = data.get("gt_answer", "")
        question = data.get("question", "")
        reasoning_chain = data.get("reasoning_chain", "")
        answer = self._extract_answer_text(question, gt_answer)
        return f"<think>{reasoning_chain}</think><answer>{answer}</answer>"

class CGMapInCGMapOutTemplate(PromptTemplate):
    """Template for CGMap-In-CGMap-Out tasks."""
    def __init__(self):
        super().__init__("cgmap_in_cgmap_out")
    
    def generate_prompt(self, data: Dict) -> str:
        """Generate CGMap-In-CGMap-Out prompt."""
        question = data.get("question", "")
        cogmap_description = data.get("grounded_cogmap_description", "")
        grounded_cogmap = data.get("grounded_cogmap", "")
        
        prompt_parts = [
            CGMAP_IN_CGMAP_OUT_BACKGROUND_INSTRUCTION.replace("<cogmap_description>", cogmap_description).replace("<grounded_cogmap>", grounded_cogmap),
            self.format_question(question)
        ]
        
        return "\n".join(prompt_parts)
    
    def generate_output(self, data: Dict) -> str:
        """Generate target output for CGMap-In-CGMap-Out."""
        gt_answer = data.get("gt_answer", "")
        question = data.get("question", "")
        grounded_cogmap = data.get("grounded_cogmap", "")
        answer = self._extract_answer_text(question, gt_answer)
        
        return f"Based on my observation, the answer is:\n<cogmap>```json\n{grounded_cogmap}\n```</cogmap><answer>{answer}</answer>"
    
    
class NoFormatFFRSNTemplate(PromptTemplate):
    """Template for NoFormat-FF-RSN (free form reasoning)."""
    def __init__(self):
        super().__init__("noformat_ff_rsn")
    
    def generate_prompt(self, data: Dict) -> str:
        """Generate NoFormat-FF-RSN prompt."""
        question = data.get("question", "")
        
        prompt_parts = [
            NOFORMAT_FF_RSN_BACKGROUND_INSTRUCTION,
            self.format_question(question)
        ]
        
        return "\n".join(prompt_parts)
    
    def generate_output(self, data: Dict) -> str:
        """Generate target output for NoFormat-FF-RSN."""
        gt_answer = data.get("gt_answer", "")
        question = data.get("question", "")
        reasoning_chain = data.get("reasoning_chain", "")
        answer = self._extract_answer_text(question, gt_answer)
        
        return f"{reasoning_chain}And my answer is: {answer}"


class NLCGMapInFFROutTemplate(PromptTemplate):
    """Template for NL-CGMap-In-FFR-Out tasks using natural language cognitive maps."""
    def __init__(self):
        super().__init__("nl_cgmap_in_ffr_out")
    
    def generate_prompt(self, data: Dict) -> str:
        """Generate NL-CGMap-In-FFR-Out prompt."""
        question = data.get("question", "")
        cogmap_description = data.get("grounded_cogmap_description", NL_GROUNDED_CGMAP_DESCRIPTION)
        grounded_cogmap = data.get("grounded_cogmap", "")
        
        # Convert JSON cogmap to natural language if needed
        nl_cogmap = convert_aug_cogmap_to_nl_cogmap(grounded_cogmap)
        
        prompt_parts = [
            NL_CGMAP_IN_FFR_OUT_BACKGROUND_INSTRUCTION.replace("<cogmap_description>", cogmap_description).replace("<grounded_cogmap>", nl_cogmap),
            self.format_question(question)
        ]
        
        return "\n".join(prompt_parts)
    
    def generate_output(self, data: Dict) -> str:
        """Generate target output for NL-CGMap-In-FFR-Out."""
        gt_answer = data.get("gt_answer", "")
        question = data.get("question", "")
        reasoning_chain = data.get("reasoning_chain", "")
        answer = self._extract_answer_text(question, gt_answer)
        
        return f"<think>{reasoning_chain}</think><answer>{answer}</answer>"


class NLAugCGMapFFROutTemplate(PromptTemplate):
    """Template for NL-Aug-CGMap-FFR-Out tasks using natural language cognitive maps."""
    def __init__(self):
        super().__init__("nl_aug_cgmap_ffr_out")
    
    def generate_prompt(self, data: Dict) -> str:
        """Generate NL-Aug-CGMap-FFR-Out prompt."""
        question = data.get("question", "")
        aug_cogmap_gen_instruction = data.get("aug_cogmap_gen_instruction", NL_AUG_CGMAP_GEN_INSTRUCTION)
        
        prompt_parts = [
            NL_AUG_CGMAP_FFR_OUT_BACKGROUND_INSTRUCTION.replace("<replace_here>", aug_cogmap_gen_instruction),
            self.format_question(question)
        ]
        
        return "\n".join(prompt_parts)
    
    def generate_output(self, data: Dict) -> str:
        """Generate target output for NL-Aug-CGMap-FFR-Out."""
        gt_answer = data.get("gt_answer", "")
        question = data.get("question", "")
        reasoning_chain = data.get("reasoning_chain", "")
        grounded_cogmap = data.get("grounded_cogmap", "")
        answer = self._extract_answer_text(question, gt_answer)
        
        # Convert JSON cogmap to natural language
        nl_cogmap = convert_aug_cogmap_to_nl_cogmap(grounded_cogmap)
        
        return f"Based on my observation, the answer is:\n<cogmap>{nl_cogmap}</cogmap><think>{reasoning_chain}</think><answer>{answer}</answer>"


class NLPlainCGMapFFROutTemplate(PromptTemplate):
    """Template for NL-Plain-CGMap-FFR-Out tasks using natural language cognitive maps."""
    def __init__(self):
        super().__init__("nl_plain_cgmap_ffr_out")
    
    def generate_prompt(self, data: Dict) -> str:
        """Generate NL-Plain-CGMap-FFR-Out prompt."""
        question = data.get("question", "")
        plain_cogmap_gen_instruction = data.get("plain_cogmap_gen_instruction", NL_PLAIN_CGMAP_GEN_INSTRUCTION)
        
        prompt_parts = [
            NL_PLAIN_CGMAP_FFR_OUT_BACKGROUND_INSTRUCTION.replace("<replace_here>", plain_cogmap_gen_instruction),
            self.format_question(question)
        ]
        
        return "\n".join(prompt_parts)
    
    def generate_output(self, data: Dict) -> str:
        """Generate target output for NL-Plain-CGMap-FFR-Out."""
        gt_answer = data.get("gt_answer", "")
        question = data.get("question", "")
        reasoning_chain = data.get("reasoning_chain", "")
        aug_grounded_cogmap = data.get("grounded_cogmap", "")
        answer = self._extract_answer_text(question, gt_answer)
        
        # Convert augmented cogmap to plain format, then to natural language
        plain_cogmap = self._convert_to_plain_cogmap(aug_grounded_cogmap)
        nl_cogmap = convert_plain_cogmap_to_nl_cogmap(plain_cogmap)
        
        return f"Based on my observation, the answer is:\n<cogmap>{nl_cogmap}</cogmap><think>{reasoning_chain}</think><answer>{answer}</answer>"
    
    def _convert_to_plain_cogmap(self, grounded_cogmap: str) -> str:
        """Convert the grounded augmented cogmap to a plain cogmap."""
        try:
            # Parse the augmented cogmap JSON
            aug_data = json.loads(grounded_cogmap)
            
            # Manually build the plain cogmap JSON string
            json_lines = ["{"]
            
            # Convert objects array to plain format
            if "objects" in aug_data:
                object_entries = []
                for obj in aug_data["objects"]:
                    obj_name = obj.get("name", "")
                    position = obj.get("position", [0, 0])
                    
                    # Build position string
                    pos_str = f"[{position[0]}, {position[1]}]"
                    
                    # Build object entry
                    if "facing" in obj:
                        facing = obj["facing"]
                        entry = f'    "{obj_name}": {{"position": {pos_str}, "facing": "{facing}"}}'
                    else:
                        entry = f'    "{obj_name}": {{"position": {pos_str}}}'
                    
                    object_entries.append(entry)
                
                # Join entries with commas
                for i, entry in enumerate(object_entries):
                    if i < len(object_entries) - 1:
                        json_lines.append(entry + ",")
                    else:
                        json_lines.append(entry)
            
            json_lines.append("}")
            
            return "\n".join(json_lines)
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Return empty object if parsing fails
            print(f"Error converting to plain cogmap: {e}")
            return "{}"

# Template registry
TEMPLATE_REGISTRY = {
    "raw_qa": RawQATemplate(),
    "ff_rsn": FFRSNTemplate(),
    "aug_cgmap_in": AugCGMapInTemplate(),
    "aug_cgmap_out": AugCGMapOutTemplate(),
    "plain_cgmap_out": PlainCGMapOutTemplate(),
    "plain_cgmap_ffr_out": PlainCGMapFFROutTemplate(),
    "aug_cgmap_ffr_out": AugCGMapFFROutTemplate(),
    "cgmap_in_ffr_out": CGMapInFFROutTemplate(),
    "cgmap_in_cgmap_out": CGMapInCGMapOutTemplate(),
    "noformat_ff_rsn": NoFormatFFRSNTemplate(),
    "nl_cgmap_in_ffr_out": NLCGMapInFFROutTemplate(),
    "nl_aug_cgmap_ffr_out": NLAugCGMapFFROutTemplate(),
    "nl_plain_cgmap_ffr_out": NLPlainCGMapFFROutTemplate(),
}


def get_template(template_name: str) -> PromptTemplate:
    """Get a template by name."""
    if template_name not in TEMPLATE_REGISTRY:
        raise ValueError(f"Unknown template: {template_name}. Available: {list(TEMPLATE_REGISTRY.keys())}")
    
    return TEMPLATE_REGISTRY[template_name]


def list_templates() -> List[str]:
    """List all available template names."""
    return list(TEMPLATE_REGISTRY.keys()) 



def test_new_templates():
    """Test function to demonstrate all new template outputs."""
    
    # Sample data for testing
    test_question = "Where is the red chair relative to the blue sofa? A. Left B. Right C. Above D. Below"
    test_reasoning = "Looking at the spatial layout, I can observe the positions of both objects. The red chair is positioned at coordinates [2, 3] while the blue sofa is at [0, 4]. When comparing their x-coordinates, the chair (x=2) is to the right of the sofa (x=0) on the grid."
    
    aug_cogmap_json = '''{
        "objects": [
            {"name": "red_chair", "position": [2, 3], "facing": "north"},
            {"name": "blue_sofa", "position": [0, 4], "facing": "east"},
            {"name": "wooden_table", "position": [5, 1]},
            {"name": "TV", "position": [7, 2]}
        ],
        "views": [
            {"name": "View 1", "position": [1, 0], "facing": "north"},
            {"name": "View 2", "position": [8, 3], "facing": "west"}
        ]
    }'''
    
    sample_data = {
        "question": test_question,
        "gt_answer": "B",
        "reasoning_chain": test_reasoning,
        "grounded_cogmap": aug_cogmap_json,
        "grounded_cogmap_description": NL_GROUNDED_CGMAP_DESCRIPTION
    }
    
    print("🔬 NEW TEMPLATE TESTING SUITE")
    print("=" * 60)
    
    # Test 1: NoFormatFFRSNTemplate
    print("\n📝 TEST 1: NoFormat-FF-RSN Template")
    print("-" * 40)
    template1 = get_template("noformat_ff_rsn")
    output1 = template1.generate_output(sample_data)
    print("Output:")
    print(output1)
    
    # Test 2: NLCGMapInFFROutTemplate
    print("\n🗺️  TEST 2: NL-CGMap-In-FFR-Out Template")
    print("-" * 40)
    template2 = get_template("nl_cgmap_in_ffr_out")
    output2 = template2.generate_output(sample_data)
    print("Output:")
    print(output2)
    
    # Test 3: NLAugCGMapFFROutTemplate
    print("\n🎯 TEST 3: NL-Aug-CGMap-FFR-Out Template")
    print("-" * 40)
    template3 = get_template("nl_aug_cgmap_ffr_out")
    output3 = template3.generate_output(sample_data)
    print("Output:")
    print(output3)
    
    # Test 4: NLPlainCGMapFFROutTemplate
    print("\n📋 TEST 4: NL-Plain-CGMap-FFR-Out Template")
    print("-" * 40)
    template4 = get_template("nl_plain_cgmap_ffr_out")
    output4 = template4.generate_output(sample_data)
    print("Output:")
    print(output4)
    
    print("\n" + "=" * 60)
    print("✅ All template tests completed!")
    

if __name__ == "__main__":
    # Test the conversion functions
    aug_cogmap = '''{
        "objects": [
            {"name": "red_chair", "position": [2, 3], "facing": "north"},
            {"name": "wooden_table", "position": [5, 1]},
            {"name": "blue_sofa", "position": [0, 4], "facing": "east"},
            {"name": "lamp", "position": [7, 2]}
        ],
        "views": [
            {"name": "View 1", "position": [1, 0], "facing": "north"},
            {"name": "View 2", "position": [8, 3], "facing": "west"},
            {"name": "View 3", "position": [4, 6], "facing": "south"}
        ]
    }'''
    
    plain_cogmap = '''{
        "dining_table": {"position": [3, 2], "facing": "south"},
        "office_chair": {"position": [3, 1]},
        "bookshelf": {"position": [0, 0], "facing": "east"},
        "computer": {"position": [2, 1], "facing": "north"},
        "plant": {"position": [5, 3]}
    }'''
    
    print("=== Conversion Functions Test ===")
    print("Augmented Cognitive Map Conversion:")
    print(convert_aug_cogmap_to_nl_cogmap(aug_cogmap))
    print("\nPlain Cognitive Map Conversion:")
    print(convert_plain_cogmap_to_nl_cogmap(plain_cogmap))
    
    print("\n" + "=" * 60)
    
    # Test new templates
    test_new_templates()