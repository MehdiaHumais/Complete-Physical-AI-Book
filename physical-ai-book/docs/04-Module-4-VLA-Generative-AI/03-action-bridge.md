# Action Bridge: Converting Natural Language to ROS 2 Commands

## Prerequisites

Before diving into this module, students should have:
- Advanced understanding of ROS 2 architecture and message types
- Experience with JSON parsing and data validation
- Knowledge of OpenAI Function Calling API
- Understanding of robot control systems and velocity commands
- Familiarity with safety systems and validation mechanisms
- Experience with Python programming and asynchronous operations

## JSON to ROS: Converting LLM Text Output to ROS 2 Messages

The conversion from LLM-generated JSON to ROS 2 messages requires careful validation and type safety to ensure reliable robot operation. This process bridges high-level AI reasoning with low-level robot control systems.

### Message Schema Validation

The conversion process involves validating the LLM's JSON output against predefined ROS message schemas:

```python
import json
from typing import Dict, Any, Union
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import String
from builtin_interfaces.msg import Time
import numpy as np

class JSONToROSConverter:
    def __init__(self):
        self.message_schemas = {
            'Twist': {
                'linear': {
                    'x': float,
                    'y': float, 
                    'z': float
                },
                'angular': {
                    'x': float,
                    'y': float,
                    'z': float
                }
            },
            'Vector3': {
                'x': float,
                'y': float,
                'z': float
            }
        }
    
    def validate_schema(self, json_data: Dict[str, Any], message_type: str) -> bool:
        """Validate JSON data against ROS message schema"""
        if message_type not in self.message_schemas:
            raise ValueError(f"Unsupported message type: {message_type}")
        
        schema = self.message_schemas[message_type]
        return self._validate_recursive(json_data, schema)
    
    def _validate_recursive(self, data: Any, schema: Dict[str, Any]) -> bool:
        """Recursively validate data against schema"""
        if isinstance(schema, dict):
            if not isinstance(data, dict):
                return False
            
            for key, expected_type in schema.items():
                if key not in data:
                    return False
                if not self._validate_recursive(data[key], expected_type):
                    return False
            return True
        elif isinstance(schema, type):
            return isinstance(data, schema)
        else:
            return False
    
    def json_to_ros_message(self, json_data: Dict[str, Any], message_type: str):
        """Convert validated JSON to ROS message"""
        if not self.validate_schema(json_data, message_type):
            raise ValueError(f"JSON data doesn't match schema for {message_type}")
        
        if message_type == 'Twist':
            return self._create_twist_message(json_data)
        elif message_type == 'Vector3':
            return self._create_vector3_message(json_data)
        else:
            raise ValueError(f"Unsupported message type: {message_type}")
    
    def _create_twist_message(self, json_data: Dict[str, Any]) -> Twist:
        """Create Twist message from JSON data"""
        msg = Twist()
        
        # Linear velocities
        linear_data = json_data.get('linear', {})
        msg.linear.x = float(linear_data.get('x', 0.0))
        msg.linear.y = float(linear_data.get('y', 0.0))
        msg.linear.z = float(linear_data.get('z', 0.0))
        
        # Angular velocities
        angular_data = json_data.get('angular', {})
        msg.angular.x = float(angular_data.get('x', 0.0))
        msg.angular.y = float(angular_data.get('y', 0.0))
        msg.angular.z = float(angular_data.get('z', 0.0))
        
        return msg
    
    def _create_vector3_message(self, json_data: Dict[str, Any]) -> Vector3:
        """Create Vector3 message from JSON data"""
        msg = Vector3()
        msg.x = float(json_data.get('x', 0.0))
        msg.y = float(json_data.get('y', 0.0))
        msg.z = float(json_data.get('z', 0.0))
        return msg
```

### Type Safety and Error Handling

```python
import logging
from decimal import Decimal, InvalidOperation

class SafeJSONConverter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def safe_float_conversion(self, value: Any, default: float = 0.0) -> float:
        """Safely convert value to float with bounds checking"""
        try:
            # Handle string numbers
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    return default
                # Use Decimal for precise conversion
                decimal_val = Decimal(value)
                float_val = float(decimal_val)
            else:
                float_val = float(value)
            
            # Check for valid float values
            if np.isnan(float_val) or np.isinf(float_val):
                self.logger.warning(f"Invalid float value detected: {float_val}, using default {default}")
                return default
            
            # Apply reasonable bounds
            max_vel = 10.0  # Reasonable max for most robots
            if abs(float_val) > max_vel:
                self.logger.warning(f"Value {float_val} exceeds maximum {max_vel}, clipping")
                return max_vel if float_val > 0 else -max_vel
            
            return float_val
            
        except (ValueError, TypeError, InvalidOperation) as e:
            self.logger.error(f"Error converting {value} to float: {e}")
            return default
    
    def safe_json_parse(self, text: str) -> Dict[str, Any]:
        """Safely parse JSON from LLM output"""
        try:
            # Clean the text to extract JSON
            json_str = self._extract_json(text)
            if not json_str:
                raise ValueError("No JSON found in text")
            
            parsed = json.loads(json_str)
            if not isinstance(parsed, dict):
                raise ValueError("Parsed JSON is not a dictionary")
            
            return parsed
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"JSON parsing error: {e}")
            raise
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that may contain other content"""
        # Look for JSON between curly braces
        start = text.find('{')
        end = text.rfind('}')
        
        if start != -1 and end != -1 and start < end:
            return text[start:end+1]
        
        # Try to find JSON array as well
        start = text.find('[')
        end = text.rfind(']')
        
        if start != -1 and end != -1 and start < end:
            return text[start:end+1]
        
        return ""
```

## Function Calling: OpenAI Function Calling for Robot Skills

OpenAI's Function Calling API provides a structured way to invoke specific robot capabilities based on natural language input, offering more reliable command execution than text parsing alone.

### Function Definition Schema

```python
def get_robot_functions():
    """Define available robot functions for OpenAI Function Calling"""
    return [
        {
            "name": "move_robot",
            "description": "Move the robot with linear and angular velocities",
            "parameters": {
                "type": "object",
                "properties": {
                    "linear_x": {
                        "type": "number",
                        "description": "Linear velocity in X direction (m/s), range: -1.0 to 1.0",
                        "minimum": -1.0,
                        "maximum": 1.0
                    },
                    "linear_y": {
                        "type": "number", 
                        "description": "Linear velocity in Y direction (m/s), range: -1.0 to 1.0",
                        "minimum": -1.0,
                        "maximum": 1.0
                    },
                    "angular_z": {
                        "type": "number",
                        "description": "Angular velocity around Z axis (rad/s), range: -2.0 to 2.0", 
                        "minimum": -2.0,
                        "maximum": 2.0
                    }
                },
                "required": ["linear_x"]
            }
        },
        {
            "name": "grasp_object",
            "description": "Grasp an object with the robot's gripper",
            "parameters": {
                "type": "object",
                "properties": {
                    "object_name": {
                        "type": "string",
                        "description": "Name of the object to grasp"
                    },
                    "grasp_type": {
                        "type": "string",
                        "enum": ["power", "precision"],
                        "description": "Type of grasp to use"
                    },
                    "force_limit": {
                        "type": "number",
                        "description": "Maximum force to apply during grasp (N)",
                        "minimum": 1.0,
                        "maximum": 50.0
                    }
                },
                "required": ["object_name"]
            }
        },
        {
            "name": "navigate_to",
            "description": "Navigate the robot to a specific location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Target location name or coordinates"
                    },
                    "speed": {
                        "type": "number",
                        "description": "Navigation speed multiplier",
                        "minimum": 0.1,
                        "maximum": 1.0
                    }
                },
                "required": ["location"]
            }
        }
    ]

class OpenAIFunctionCaller:
    def __init__(self, api_key: str):
        import openai
        openai.api_key = api_key
        self.functions = get_robot_functions()
    
    async def call_function(self, user_input: str, robot_state: Dict[str, Any]) -> Dict[str, Any]:
        """Call appropriate robot function based on user input"""
        import openai
        
        messages = [
            {
                "role": "system", 
                "content": f"You are a robot assistant. Current robot state: {robot_state}. "
                          "Choose the most appropriate function to accomplish the user's request."
            },
            {
                "role": "user",
                "content": user_input
            }
        ]
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo-0613",  # Function Calling model
            messages=messages,
            functions=self.functions,
            function_call="auto"
        )
        
        message = response.choices[0].message
        
        if message.get("function_call"):
            function_name = message["function_call"]["name"]
            function_args = json.loads(message["function_call"]["arguments"])
            
            return {
                "function_name": function_name,
                "function_args": function_args,
                "success": True
            }
        else:
            return {
                "function_name": None,
                "function_args": {},
                "success": False,
                "message": "No function call generated"
            }
```

## Complete Implementation: llm_bridge_node.py

Here's a complete ROS 2 node that bridges LLM output to robot commands:

```python
#!/usr/bin/env python3
"""
LLM Bridge Node: Converts natural language commands to ROS 2 velocity commands
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
import json
import threading
import time
import numpy as np
from typing import Dict, Any, Optional


class LLMBridgeNode(Node):
    def __init__(self):
        super().__init__('llm_bridge_node')
        
        # Initialize JSON converter
        self.converter = SafeJSONConverter()
        
        # Create publishers
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.velocity_publisher = self.create_publisher(
            Twist,
            'cmd_vel',
            qos_profile
        )
        
        # Create subscribers
        self.command_subscriber = self.create_subscription(
            String,
            'llm_commands',
            self.command_callback,
            qos_profile
        )
        
        self.laser_subscriber = self.create_subscription(
            LaserScan,
            'scan',
            self.laser_callback,
            qos_profile
        )
        
        # State variables
        self.last_command_time = time.time()
        self.safety_enabled = True
        self.current_velocity = Twist()
        self.laser_data = None
        
        # Command timeout (stop if no new commands for this duration)
        self.command_timeout = 5.0  # seconds
        
        # Start safety monitoring timer
        self.safety_timer = self.create_timer(0.1, self.safety_check)
        
        self.get_logger().info('LLM Bridge Node initialized')
    
    def laser_callback(self, msg: LaserScan):
        """Update laser scan data for safety checks"""
        self.laser_data = msg
    
    def command_callback(self, msg: String):
        """Process incoming LLM commands"""
        self.last_command_time = time.time()
        
        try:
            # Parse and validate the command
            parsed_command = self.converter.safe_json_parse(msg.data)
            
            # Validate against safety constraints
            if not self.is_command_safe(parsed_command):
                self.get_logger().warn('Unsafe command blocked')
                return
            
            # Convert to Twist message
            twist_msg = self.converter.json_to_ros_message(parsed_command, 'Twist')
            
            # Publish the command
            self.velocity_publisher.publish(twist_msg)
            self.current_velocity = twist_msg
            
            self.get_logger().info(
                f'Published velocity command: linear=({twist_msg.linear.x:.2f}, '
                f'{twist_msg.linear.y:.2f}, {twist_msg.linear.z:.2f}), '
                f'angular=({twist_msg.angular.x:.2f}, {twist_msg.angular.y:.2f}, '
                f'{twist_msg.angular.z:.2f})'
            )
            
        except Exception as e:
            self.get_logger().error(f'Error processing command: {e}')
            # Publish zero velocity on error
            self.publish_stop_command()
    
    def is_command_safe(self, command: Dict[str, Any]) -> bool:
        """Check if the command is safe to execute"""
        if not self.safety_enabled:
            return True
        
        # Check linear velocity limits
        linear = command.get('linear', {})
        linear_x = linear.get('x', 0.0)
        linear_y = linear.get('y', 0.0)
        
        # Maximum safe linear velocity
        max_linear = 1.0  # m/s
        if abs(linear_x) > max_linear or abs(linear_y) > max_linear:
            self.get_logger().warn(f'Linear velocity exceeds safe limit: {linear_x}, {linear_y}')
            return False
        
        # Check for collision risk based on laser data
        if self.laser_data and (abs(linear_x) > 0.1 or abs(linear_y) > 0.1):
            min_distance = min(self.laser_data.ranges) if self.laser_data.ranges else float('inf')
            safe_distance = 0.5  # meters
            
            if min_distance < safe_distance:
                self.get_logger().warn(
                    f'Collision risk detected: distance {min_distance:.2f}m < safe {safe_distance}m'
                )
                return False
        
        return True
    
    def safety_check(self):
        """Safety timer callback to stop robot if needed"""
        time_since_last_command = time.time() - self.last_command_time
        
        # Stop robot if no commands received within timeout
        if time_since_last_command > self.command_timeout:
            self.publish_stop_command()
            self.get_logger().warn('Command timeout - robot stopped for safety')
    
    def publish_stop_command(self):
        """Publish zero velocity to stop the robot"""
        stop_msg = Twist()
        self.velocity_publisher.publish(stop_msg)
        self.current_velocity = stop_msg


def main(args=None):
    rclpy.init(args=args)
    
    # Check for required parameters
    node = LLMBridgeNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop_command()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Launch File Configuration

```xml
<launch>
  <!-- LLM Bridge Node -->
  <node pkg="llm_bridge" exec="llm_bridge_node.py" name="llm_bridge_node">
    <param name="command_timeout" value="5.0"/>
    <param name="safety_enabled" value="true"/>
  </node>
  
  <!-- Example: Bridge between text topic and velocity commands -->
  <node pkg="llm_bridge" exec="command_parser.py" name="command_parser">
    <remap from="llm_input" to="/chatbot_response"/>
    <remap from="llm_commands" to="/llm_bridge_node/llm_commands"/>
  </node>
</launch>
```

## Safety: Implementing Guardrails and Validation

Safety systems are critical when bridging AI systems to physical robots. The following guardrails prevent dangerous or unintended robot behavior.

### Multi-level Safety Architecture

```python
class SafetyGuardrails:
    def __init__(self, node: Node):
        self.node = node
        self.emergency_stop = False
        self.safety_limits = {
            'max_linear_velocity': 1.0,      # m/s
            'max_angular_velocity': 2.0,     # rad/s
            'max_acceleration': 2.0,         # m/s²
            'min_distance_obstacle': 0.3,    # meters
            'max_command_frequency': 10.0    # Hz
        }
        
        # Previous command state for velocity limiting
        self.prev_command_time = time.time()
        self.prev_velocity = Twist()
    
    def validate_command(self, twist_msg: Twist, laser_scan: Optional[LaserScan] = None) -> bool:
        """Validate command against all safety constraints"""
        checks = [
            self._check_velocity_limits(twist_msg),
            self._check_acceleration_limits(twist_msg),
            self._check_collision_risk(twist_msg, laser_scan),
            self._check_command_frequency()
        ]
        
        return all(checks)
    
    def _check_velocity_limits(self, twist_msg: Twist) -> bool:
        """Check if velocity commands are within safe limits"""
        max_lin = self.safety_limits['max_linear_velocity']
        max_ang = self.safety_limits['max_angular_velocity']
        
        if (abs(twist_msg.linear.x) > max_lin or 
            abs(twist_msg.linear.y) > max_lin or 
            abs(twist_msg.linear.z) > max_lin):
            self.node.get_logger().warn(f'Linear velocity exceeds limit: {twist_msg.linear}')
            return False
        
        if (abs(twist_msg.angular.x) > max_ang or 
            abs(twist_msg.angular.y) > max_ang or 
            abs(twist_msg.angular.z) > max_ang):
            self.node.get_logger().warn(f'Angular velocity exceeds limit: {twist_msg.angular}')
            return False
        
        return True
    
    def _check_acceleration_limits(self, twist_msg: Twist) -> bool:
        """Check if acceleration is within safe limits"""
        current_time = time.time()
        time_delta = current_time - self.prev_command_time
        
        if time_delta > 0:
            # Calculate acceleration
            linear_acc = abs(twist_msg.linear.x - self.prev_velocity.linear.x) / time_delta
            angular_acc = abs(twist_msg.angular.z - self.prev_velocity.angular.z) / time_delta
            
            max_acc = self.safety_limits['max_acceleration']
            
            if linear_acc > max_acc or angular_acc > max_acc:
                self.node.get_logger().warn(
                    f'Acceleration exceeds limit: linear={linear_acc:.2f}, '
                    f'angular={angular_acc:.2f} (max={max_acc})'
                )
                return False
        
        # Update state
        self.prev_command_time = current_time
        self.prev_velocity = twist_msg
        
        return True
    
    def _check_collision_risk(self, twist_msg: Twist, laser_scan: Optional[LaserScan]) -> bool:
        """Check for collision risk based on laser data"""
        if not laser_scan or (twist_msg.linear.x <= 0 and twist_msg.linear.y <= 0):
            return True  # Only check when moving forward
        
        # Check forward direction for obstacles
        min_range = min(laser_scan.ranges) if laser_scan.ranges else float('inf')
        safe_range = self.safety_limits['min_distance_obstacle']
        
        if min_range < safe_range:
            self.node.get_logger().warn(
                f'Collision risk: obstacle at {min_range:.2f}m (safe: {safe_range}m)'
            )
            return False
        
        return True
    
    def _check_command_frequency(self) -> bool:
        """Check if commands are coming at safe frequency"""
        # This would be implemented by tracking command arrival rates
        # and ensuring they're within the robot's control capabilities
        return True
    
    def emergency_stop(self):
        """Trigger emergency stop"""
        self.emergency_stop = True
        self.node.get_logger().error('EMERGENCY STOP ACTIVATED')
        # Additional emergency procedures would go here
```

### Integration with Navigation Stack

```python
class SafeLLMBridgeNode(LLMBridgeNode):
    def __init__(self):
        super().__init__()
        
        # Initialize safety guardrails
        self.safety_guardrails = SafetyGuardrails(self)
        
        # Create action clients for navigation stack
        from nav2_msgs.action import NavigateToPose
        from rclpy.action import ActionClient
        
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # Override the command callback with safety validation
        self.command_subscriber = self.create_subscription(
            String,
            'llm_commands',
            self.safe_command_callback,
            10
        )
    
    def safe_command_callback(self, msg: String):
        """Process commands with safety validation"""
        try:
            # Parse the command
            parsed_command = self.converter.safe_json_parse(msg.data)
            
            # Convert to Twist message
            twist_msg = self.converter.json_to_ros_message(parsed_command, 'Twist')
            
            # Validate with safety guardrails
            is_safe = self.safety_guardrails.validate_command(twist_msg, self.laser_data)
            
            if is_safe:
                # Publish the command
                self.velocity_publisher.publish(twist_msg)
                self.current_velocity = twist_msg
            else:
                self.get_logger().warn('Command failed safety validation')
                self.publish_stop_command()
                
        except Exception as e:
            self.get_logger().error(f'Safety error processing command: {e}')
            self.publish_stop_command()
```

### Command Validation Pipeline

```python
class CommandValidationPipeline:
    def __init__(self):
        self.validators = [
            self._syntax_validator,
            self._semantic_validator,
            self._safety_validator,
            self._context_validator
        ]
    
    def validate_command(self, command: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate command through multiple validation stages"""
        results = {
            'valid': True,
            'errors': [],
            'parsed_command': None
        }
        
        try:
            # Parse JSON
            parsed = json.loads(command)
            results['parsed_command'] = parsed
            
            # Run all validators
            for validator in self.validators:
                is_valid, error = validator(parsed, context)
                if not is_valid:
                    results['valid'] = False
                    results['errors'].append(error)
                    
        except json.JSONDecodeError as e:
            results['valid'] = False
            results['errors'].append(f"JSON parsing error: {str(e)}")
        
        return results
    
    def _syntax_validator(self, command: Dict, context: Dict) -> tuple[bool, str]:
        """Validate command syntax"""
        required_fields = ['linear', 'angular']
        for field in required_fields:
            if field not in command:
                return False, f"Missing required field: {field}"
        return True, ""
    
    def _semantic_validator(self, command: Dict, context: Dict) -> tuple[bool, str]:
        """Validate command semantics"""
        # Check if command makes sense for current robot
        return True, ""
    
    def _safety_validator(self, command: Dict, context: Dict) -> tuple[bool, str]:
        """Validate safety constraints"""
        # Check velocity limits
        linear = command.get('linear', {})
        if abs(linear.get('x', 0)) > 1.0:
            return False, "Linear velocity too high"
        return True, ""
    
    def _context_validator(self, command: Dict, context: Dict) -> tuple[bool, str]:
        """Validate based on context"""
        # Check if command is appropriate given current state
        robot_state = context.get('robot_state', {})
        if robot_state.get('battery_level', 100) < 10 and command.get('linear', {}).get('x', 0) != 0:
            return False, "Low battery - movement not allowed"
        return True, ""
```

## Summary

This comprehensive software engineering guide has detailed the complete pipeline for converting natural language commands from large language models into safe, executable ROS 2 robot commands. We've explored the critical process of JSON to ROS message conversion with proper schema validation and type safety, implemented OpenAI Function Calling for structured robot control, provided a complete ROS 2 bridge node implementation, and detailed comprehensive safety guardrails and validation mechanisms.

The guide emphasizes the critical importance of safety when bridging AI systems to physical robots, demonstrating multiple validation layers and emergency procedures. The modular design allows for easy integration with existing ROS 2 systems while maintaining the reliability and safety required for real-world robotic applications.

Understanding these concepts is essential for developing production-ready robotic systems that can safely interpret and execute commands from large language models while maintaining the safety and reliability standards required in physical environments.