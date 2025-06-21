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
}


def get_template(template_name: str) -> PromptTemplate:
    """Get a template by name."""
    if template_name not in TEMPLATE_REGISTRY:
        raise ValueError(f"Unknown template: {template_name}. Available: {list(TEMPLATE_REGISTRY.keys())}")
    
    return TEMPLATE_REGISTRY[template_name]


def list_templates() -> List[str]:
    """List all available template names."""
    return list(TEMPLATE_REGISTRY.keys()) 