# Cognitive Planning: Prompt Engineering for LLM-Based Robot Control

## Prerequisites

Before diving into this module, students should have:
- Understanding of large language model architectures and capabilities
- Experience with prompt engineering and natural language processing
- Knowledge of robotic task planning and decomposition
- Familiarity with ROS 2 action and service interfaces
- Understanding of cognitive architecture concepts
- Basic knowledge of state machines and behavior trees

## The Prefrontal Cortex: LLMs as High-Level Planners

The prefrontal cortex in human cognition serves as the executive control center, responsible for planning, decision-making, and coordinating complex behaviors. In robotic systems, large language models (LLMs) can function analogously as high-level cognitive planners that decompose complex goals into executable action sequences.

### Cognitive Architecture for Robotics

The cognitive planning architecture mirrors human cognitive processes:

1. **Goal Processing**: Understanding high-level commands and intentions
2. **Plan Generation**: Decomposing goals into sub-goals and atomic actions
3. **Context Integration**: Incorporating environmental and situational awareness
4. **Action Selection**: Choosing optimal actions based on current state
5. **Monitoring and Adaptation**: Adjusting plans based on feedback

### LLM as Executive Controller

The LLM serves as the executive controller in the cognitive architecture by:

$$\text{Plan} = \text{LLM}(\text{Goal}, \text{State}, \text{Context})$$

Where the LLM processes:
- **Goal**: High-level task specification
- **State**: Current robot and environment state
- **Context**: Available tools, constraints, and preferences

### Mathematical Framework for Cognitive Planning

The planning process can be formalized as:

$$\pi^* = \arg\max_{\pi} \mathbb{E}[R(\pi, s_0, G)]$$

Where:
- $\pi$ is the plan sequence
- $s_0$ is the initial state
- $G$ is the goal specification
- $R$ is the reward function incorporating task completion and efficiency

The LLM approximates this optimization through in-context learning and pattern recognition.

## Chain of Thought: Prompting Strategies for Robotic Reasoning

Chain of Thought (CoT) prompting enables LLMs to decompose complex reasoning tasks into step-by-step components, crucial for robotic planning where sequential action execution is required.

### CoT Prompting Framework

The CoT approach for robotics follows a structured format:

```
[GOAL]: {high_level_task}
[CONTEXT]: {current_state, available_actions, constraints}
[REASONING]: Think step by step:
1. Analyze the goal and identify sub-goals
2. Consider current state and available resources
3. Plan the sequence of actions
4. Identify potential obstacles and alternatives
[OUTPUT]: {structured_action_sequence}
```

### Step-by-Step Reasoning Examples

```python
def create_cot_prompt(goal, current_state, available_actions):
    """
    Create Chain of Thought prompt for robotic planning
    """
    prompt = f"""
You are a robot planning system. Your task is to decompose high-level goals into specific, executable actions.

[GOAL]: {goal}
[STATE]: {current_state}
[AVAILABLE ACTIONS]: {available_actions}

Please think step by step to decompose this goal:

Step 1: Analyze the goal and identify what needs to be accomplished
Step 2: Consider the current state and determine what information is needed
Step 3: Plan the sequence of actions to achieve the goal
Step 4: Consider potential obstacles and alternative approaches

[REASONING]:
"""
    return prompt

def create_structured_output_prompt(goal, current_state, available_actions):
    """
    Create structured output prompt with explicit format requirements
    """
    prompt = f"""
You are a robotic task planner. Convert the high-level goal into a sequence of executable actions.

GOAL: {goal}
CURRENT STATE: {current_state}
AVAILABLE ACTIONS: {available_actions}

Think step by step:
1. What is the main objective?
2. What subtasks are required?
3. What is the optimal sequence?
4. What are potential challenges?

Then provide your response in this exact format:
[THOUGHT_PROCESS]: <your reasoning here>
[SUBTASKS]:
- [action_1] <description>
- [action_2] <description>
[EXECUTION_ORDER]: <numbered sequence>

Example response format:
[THOUGHT_PROCESS]: The goal requires cleaning the kitchen. First, I need to locate cleaning supplies, then identify dirty areas, and finally execute cleaning actions.
[SUBTASKS]:
- [NAVIGATE] Find the sponge in the cleaning supplies area
- [GRASP] Pick up the cleaning sponge
- [NAVIGATE] Move to the sink to wet the sponge
- [CLEAN] Clean the dirty surface
[EXECUTION_ORDER]: 1. NAVIGATE to supplies, 2. GRASP sponge, 3. NAVIGATE to sink, 4. CLEAN surface
"""
    return prompt
```

### Advanced CoT Techniques

**Self-Consistency**: Generate multiple reasoning paths and select the most consistent:

```python
def self_consistency_planning(goal, current_state, available_actions, n_samples=3):
    """
    Generate multiple planning samples and select the most consistent
    """
    import asyncio
    import openai
    
    # Generate multiple samples
    tasks = []
    for i in range(n_samples):
        prompt = create_structured_output_prompt(goal, current_state, available_actions)
        task = openai.Completion.acreate(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=500,
            temperature=0.7
        )
        tasks.append(task)
    
    responses = await asyncio.gather(*tasks)
    
    # Analyze consistency across responses
    # This would involve comparing action sequences and selecting most common approach
    return responses
```

**Reflection and Verification**: Include reasoning verification in the prompt:

```python
def reflection_prompt(goal, current_state, available_actions):
    """
    Include reflection and verification steps in the prompt
    """
    prompt = f"""
You are a robot planning system. Your goal is to create a robust plan that accounts for potential issues.

GOAL: {goal}
CURRENT STATE: {current_state}
AVAILABLE ACTIONS: {available_actions}

THINK STEP BY STEP:
1. Analyze the goal requirements
2. Break down into subtasks
3. For each subtask, identify potential failure modes
4. Plan verification steps after each subtask
5. Plan recovery actions if subtasks fail

[REFLECTION]: Consider whether the plan is robust to:
- Object not found
- Obstacles in navigation
- Gripper failures
- Environmental changes

[VERIFICATION]: After each action, how will you confirm success?

[ROBUST PLAN]:
"""
    return prompt
```

## Task Decomposition Examples

### Kitchen Cleaning Example

Converting "Clean the kitchen" into atomic tasks:

**High-Level Goal**: "Clean the kitchen"
**Decomposed Tasks**:
1. **Perception**: Scan the kitchen to identify dirty areas
2. **Resource Location**: Find cleaning supplies (sponge, soap, etc.)
3. **Surface Preparation**: Clear surfaces of items
4. **Cleaning Execution**: Clean each identified surface
5. **Verification**: Check that surfaces meet cleanliness criteria
6. **Cleanup**: Return cleaning supplies to original location

**Detailed Decomposition**:

```python
def decompose_clean_kitchen():
    """
    Decompose 'Clean the kitchen' into executable tasks
    """
    high_level_task = "Clean the kitchen"
    
    decomposed_tasks = [
        {
            "task_id": "1",
            "action": "SCENE_UNDERSTANDING",
            "description": "Scan the kitchen environment to identify dirty surfaces, obstacles, and cleaning supplies",
            "requirements": ["navigation", "perception"],
            "expected_duration": "30s",
            "success_criteria": "Map of kitchen with identified dirty areas"
        },
        {
            "task_id": "2",
            "action": "LOCATE_CLEANING_SUPPLIES",
            "description": "Navigate to and identify cleaning supplies (sponge, cloth, cleaning solution)",
            "requirements": ["navigation", "object_recognition", "manipulation"],
            "expected_duration": "60s",
            "success_criteria": "Located and verified availability of cleaning supplies"
        },
        {
            "task_id": "3",
            "action": "GRASP_CLEANING_TOOL",
            "description": "Pick up the cleaning sponge or cloth",
            "requirements": ["manipulation", "grasping"],
            "expected_duration": "15s", 
            "success_criteria": "Sponge successfully grasped and held"
        },
        {
            "task_id": "4",
            "action": "APPLY_CLEANING_ACTION",
            "description": "Clean the identified dirty surfaces systematically",
            "requirements": ["manipulation", "navigation", "path_planning"],
            "expected_duration": "300s",
            "success_criteria": "Surfaces are clean based on visual inspection"
        },
        {
            "task_id": "5",
            "action": "VERIFY_CLEANLINESS",
            "description": "Inspect cleaned surfaces to ensure they meet cleanliness standards",
            "requirements": ["perception", "navigation"],
            "expected_duration": "60s",
            "success_criteria": "All surfaces pass cleanliness verification"
        },
        {
            "task_id": "6",
            "action": "CLEANUP",
            "description": "Return cleaning supplies to their designated locations",
            "requirements": ["navigation", "manipulation"],
            "expected_duration": "30s",
            "success_criteria": "Cleaning supplies properly stored"
        }
    ]
    
    return decomposed_tasks
```

### Other Common Task Decompositions

**"Set the table for dinner"**:
- Locate dinnerware (plates, utensils, glasses)
- Navigate to dining table
- Arrange items in proper positions
- Verify table setting completeness

**"Pour coffee in the kitchen"**:
- Navigate to coffee maker/area
- Locate coffee cup
- Grasp and position cup under coffee source
- Execute pour action
- Return cup to appropriate location

## Python Implementation: LLM Task Parser

Here's a complete Python function that sends prompts to an LLM and parses the resulting task list:

```python
#!/usr/bin/env python3
"""
LLM Task Parser: Convert natural language commands to structured task sequences
"""

import json
import re
import asyncio
import openai
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class TaskAction(str, Enum):
    """Enumeration of available robot actions"""
    NAVIGATE = "NAVIGATE"
    GRASP = "GRASP"
    RELEASE = "RELEASE"
    DETECT_OBJECT = "DETECT_OBJECT"
    MANIPULATE = "MANIPULATE"
    SPEAK = "SPEAK"
    WAIT = "WAIT"
    PERCEIVE = "PERCEIVE"


@dataclass
class Task:
    """Represents a single robot task"""
    id: str
    action: TaskAction
    description: str
    parameters: Dict[str, Any]
    expected_duration: float  # in seconds
    preconditions: List[str]
    postconditions: List[str]
    priority: int = 0


class LLMTaskParser:
    """Parses natural language commands into structured robot tasks using LLMs"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        openai.api_key = api_key
        
        # Define available actions for the robot
        self.available_actions = [
            {
                "name": "NAVIGATE",
                "description": "Move the robot to a specific location",
                "parameters": ["target_location", "speed"]
            },
            {
                "name": "GRASP",
                "description": "Pick up an object with the robot's gripper",
                "parameters": ["object_name", "grasp_pose"]
            },
            {
                "name": "DETECT_OBJECT",
                "description": "Detect and locate specific objects in the environment",
                "parameters": ["object_type", "search_area"]
            },
            {
                "name": "MANIPULATE",
                "description": "Perform manipulation actions like opening, closing, pouring",
                "parameters": ["action_type", "target_object", "parameters"]
            },
            {
                "name": "SPEAK",
                "description": "Speak a message to humans",
                "parameters": ["message"]
            },
            {
                "name": "PERCEIVE",
                "description": "Perceive and understand the current environment state",
                "parameters": ["sensor_types", "target_objects"]
            }
        ]
    
    def create_planning_prompt(self, command: str, robot_state: Dict[str, Any]) -> str:
        """Create a structured prompt for task planning"""
        prompt = f"""
You are a robot task planning system. Your job is to decompose high-level commands into specific, executable robot actions.

COMMAND: "{command}"

ROBOT STATE:
- Current location: {robot_state.get('location', 'unknown')}
- Battery level: {robot_state.get('battery_level', 'unknown')}%
- Available grippers: {robot_state.get('grippers', 'unknown')}
- Current held object: {robot_state.get('held_object', 'none')}
- Connected sensors: {robot_state.get('sensors', 'unknown')}

AVAILABLE ACTIONS:
"""
        
        for action in self.available_actions:
            prompt += f"- {action['name']}: {action['description']}\n"
            prompt += f"  Parameters: {', '.join(action['parameters'])}\n"
        
        prompt += f"""

Please think step by step:
1. Analyze the command and identify the main objective
2. Determine what subtasks are required
3. Consider the robot's current state and constraints
4. Plan the optimal sequence of actions
5. Consider error handling and verification

Provide your response in the following JSON format:
{{
    "thought_process": "<your reasoning here>",
    "task_sequence": [
        {{
            "id": "1",
            "action": "NAVIGATE",
            "description": "Move to the kitchen",
            "parameters": {{
                "target_location": "kitchen",
                "speed": 0.5
            }},
            "expected_duration": 30.0,
            "preconditions": ["robot_is_operational"],
            "postconditions": ["robot_at_kitchen"]
        }}
    ]
}}

Only respond with valid JSON, no other text.
"""
        return prompt
    
    async def parse_command_to_tasks(self, command: str, robot_state: Dict[str, Any]) -> List[Task]:
        """Parse a natural language command into a sequence of tasks"""
        try:
            prompt = self.create_planning_prompt(command, robot_state)
            
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful robot task planning assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Extract the response content
            response_text = response.choices[0].message.content.strip()
            
            # Clean up the response to extract JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_response = json.loads(json_str)
                
                # Convert to Task objects
                tasks = []
                for task_data in parsed_response.get("task_sequence", []):
                    task = Task(
                        id=task_data["id"],
                        action=TaskAction(task_data["action"]),
                        description=task_data["description"],
                        parameters=task_data.get("parameters", {}),
                        expected_duration=task_data.get("expected_duration", 30.0),
                        preconditions=task_data.get("preconditions", []),
                        postconditions=task_data.get("postconditions", []),
                        priority=int(task_data.get("id", 0))  # Use ID for priority
                    )
                    tasks.append(task)
                
                return tasks
            else:
                print(f"Could not extract JSON from response: {response_text}")
                return self._fallback_tasks(command)
                
        except json.JSONDecodeError:
            print("Failed to parse JSON response from LLM")
            return self._fallback_tasks(command)
        except Exception as e:
            print(f"Error parsing command: {e}")
            return self._fallback_tasks(command)
    
    def _fallback_tasks(self, command: str) -> List[Task]:
        """Create fallback tasks if LLM parsing fails"""
        print(f"Using fallback parsing for command: {command}")
        
        # Simple fallback based on command keywords
        if "clean" in command.lower():
            return [
                Task(
                    id="1",
                    action=TaskAction.PERCEIVE,
                    description="Scan environment to identify dirty areas",
                    parameters={"sensor_type": "camera"},
                    expected_duration=10.0,
                    preconditions=[],
                    postconditions=["environment_map_generated"]
                ),
                Task(
                    id="2", 
                    action=TaskAction.NAVIGATE,
                    description="Move to cleaning supplies location",
                    parameters={"target_location": "cleaning_station"},
                    expected_duration=30.0,
                    preconditions=["environment_map_generated"],
                    postconditions=["robot_at_cleaning_supplies"]
                )
            ]
        elif "go to" in command.lower() or "move to" in command.lower():
            # Extract location from command
            location = re.search(r'(?:go to|move to|navigate to)\s+([^.!?]+)', command, re.IGNORECASE)
            location = location.group(1).strip() if location else "unknown location"
            
            return [
                Task(
                    id="1",
                    action=TaskAction.NAVIGATE,
                    description=f"Navigate to {location}",
                    parameters={"target_location": location},
                    expected_duration=60.0,
                    preconditions=[],
                    postconditions=[f"robot_at_{location.replace(' ', '_')}"]
                )
            ]
        else:
            # Generic fallback
            return [
                Task(
                    id="1",
                    action=TaskAction.SPEAK,
                    description="Unable to parse command, please rephrase",
                    parameters={"message": "I don't understand that command. Can you please rephrase?"},
                    expected_duration=5.0,
                    preconditions=[],
                    postconditions=["command_failed"]
                )
            ]
    
    def validate_task_sequence(self, tasks: List[Task]) -> bool:
        """Validate that the task sequence is logically consistent"""
        if not tasks:
            return False
        
        # Check for valid action types
        for task in tasks:
            try:
                TaskAction(task.action)
            except ValueError:
                print(f"Invalid action type: {task.action}")
                return False
        
        # Check for circular dependencies in preconditions/postconditions
        all_postconditions = set()
        for task in tasks:
            all_postconditions.update(task.postconditions)
        
        # In a real system, you'd check actual dependencies
        # For now, just ensure basic structure is valid
        return True


def main():
    """Example usage of the LLM Task Parser"""
    import os
    
    # Initialize the task parser
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    parser = LLMTaskParser(api_key)
    
    # Example robot state
    robot_state = {
        "location": "living_room",
        "battery_level": 85,
        "grippers": ["left_arm", "right_arm"],
        "held_object": "none",
        "sensors": ["camera", "lidar", "imu", "touch_sensors"]
    }
    
    # Example commands to test
    test_commands = [
        "Clean the kitchen thoroughly",
        "Go to the kitchen and bring me a glass of water",
        "Set the table for two people in the dining room"
    ]
    
    async def process_commands():
        for command in test_commands:
            print(f"\nProcessing command: '{command}'")
            print("-" * 50)
            
            tasks = await parser.parse_command_to_tasks(command, robot_state)
            
            if parser.validate_task_sequence(tasks):
                print("Generated task sequence:")
                for i, task in enumerate(tasks, 1):
                    print(f"  {i}. {task.action.value}: {task.description}")
                    print(f"     Parameters: {task.parameters}")
                    print(f"     Expected duration: {task.expected_duration}s")
            else:
                print("Generated task sequence is invalid")
    
    # Run the async function
    asyncio.run(process_commands())


if __name__ == "__main__":
    main()
```

### Integration with ROS 2

The parsed tasks can be integrated with ROS 2 action servers:

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import String

class TaskExecutorNode(Node):
    """ROS 2 node to execute parsed tasks"""
    
    def __init__(self):
        super().__init__('task_executor')
        
        # Action clients for different robot capabilities
        self.nav_client = ActionClient(self, NavigateAction, 'navigate_to_pose')
        self.manipulation_client = ActionClient(self, ManipulateAction, 'manipulation_controller')
        
        # Task queue
        self.task_queue = []
        self.current_task_index = 0
        
    def execute_task_sequence(self, tasks: List[Task]):
        """Execute a sequence of parsed tasks"""
        self.task_queue = tasks
        self.current_task_index = 0
        self.execute_next_task()
        
    def execute_next_task(self):
        """Execute the next task in the sequence"""
        if self.current_task_index >= len(self.task_queue):
            self.get_logger().info('All tasks completed')
            return
            
        current_task = self.task_queue[self.current_task_index]
        self.get_logger().info(f'Executing task {current_task.id}: {current_task.action}')
        
        if current_task.action == TaskAction.NAVIGATE:
            self.execute_navigation_task(current_task)
        elif current_task.action == TaskAction.GRASP:
            self.execute_manipulation_task(current_task)
        # ... handle other action types
        
    def execute_navigation_task(self, task: Task):
        """Execute a navigation task"""
        goal_msg = NavigateAction.Goal()
        goal_msg.pose = self.create_pose_from_params(task.parameters)
        
        self.nav_client.wait_for_server()
        self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.navigation_feedback_callback
        ).add_done_callback(self.navigation_done_callback)
```

## Summary

This comprehensive prompt engineering guide has explored the use of large language models as cognitive planners for robotic systems, modeling them after the human prefrontal cortex's executive function. We've detailed Chain of Thought prompting strategies for robust robotic reasoning, provided concrete examples of task decomposition from high-level goals to atomic actions, and implemented a complete Python system for parsing natural language commands into structured robot tasks.

The guide demonstrates how to build a cognitive architecture that can understand complex, natural language commands and translate them into executable robotic actions. The modular design allows for easy integration with ROS 2 systems and can be extended to support more complex cognitive behaviors and reasoning patterns.

Understanding these concepts is essential for developing advanced robotic systems that can interpret human intentions through natural language and execute complex, multi-step tasks in real-world environments.