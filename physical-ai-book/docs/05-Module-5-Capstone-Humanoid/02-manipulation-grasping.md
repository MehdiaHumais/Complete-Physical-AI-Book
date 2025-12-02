# Manipulation and Grasping Control Guide

## Prerequisites

Before diving into this module, students should have:
- Understanding of robotic kinematics and inverse kinematics
- Knowledge of coordinate transformations and frames
- Experience with sensor integration and perception systems
- Familiarity with control theory and trajectory planning
- Basic understanding of grasp planning and contact mechanics
- Experience with motion planning algorithms (RRT, A*, etc.)

## End Effectors: Parallel Grippers vs Dexterous Hands

The choice of end effector significantly impacts a humanoid robot's manipulation capabilities, affecting both the complexity of grasp planning and the types of objects that can be successfully manipulated.

### Parallel Grippers

Parallel grippers feature two opposing fingers that move in parallel to grasp objects. The mechanical simplicity of parallel grippers provides several advantages:

**Mechanical Design**: The parallel motion creates a consistent grasp point regardless of object size within the gripper's range. The grasp force is applied uniformly across the contact surfaces.

**Control Simplicity**: Parallel grippers typically require only one actuator to control both fingers simultaneously, reducing control complexity. The grasp force can be regulated through current control of the motor.

**Stability**: The parallel approach provides stable grasp points that are predictable and easy to model mathematically. The grasp configuration is largely independent of object geometry within operational limits.

**Mathematical Model**: For a parallel gripper with finger span $d$ and grasp force $F_g$:

$$\mathbf{F}_{total} = \mathbf{F}_{left} + \mathbf{F}_{right}$$
$$d_{object} \leq d_{max} - \delta_{clearance}$$

Where $\delta_{clearance}$ provides safety margin for object size variations.

### Dexterous Hands

Dexterous hands, such as anthropomorphic multi-fingered hands, provide greater manipulation flexibility through multiple independently-controlled fingers.

**Degrees of Freedom**: A typical dexterous hand has 16-24 DOF, allowing for complex grasp configurations including power grasps, precision grasps, and fingertip manipulation.

**Grasp Types**: Dexterous hands can execute various grasp types:

- **Power Grasps**: Cylindrical, spherical, and hook grasps for heavy objects
- **Precision Grasps**: Tip pinch, lateral pinch, and three-finger tripod grasps for fine manipulation
- **Complex Grasps**: Multi-finger coordination for irregular objects

**Control Complexity**: Each finger has multiple joints (typically 3-4 per finger), requiring sophisticated control algorithms and grasp planning.

### Grasp Force Optimization

The grasp force must balance between sufficient grip to prevent slipping and minimal force to avoid object damage:

$$\sum_{i=1}^{n} \mathbf{f}_i \geq \mathbf{F}_{external}$$
$$\sum_{i=1}^{n} \mathbf{r}_i \times \mathbf{f}_i \geq \mathbf{M}_{external}$$

Where $\mathbf{f}_i$ is the contact force at point $i$, $\mathbf{r}_i$ is the position vector from object center of mass, and $\mathbf{F}_{external}$ and $\mathbf{M}_{external}$ are external forces and moments.

### Grasp Planning Algorithms

```python
import numpy as np
from scipy.spatial import ConvexHull
from enum import Enum

class GraspType(Enum):
    PARALLEL = "parallel"
    DEXTEROUS_POWER = "dexterous_power"
    DEXTEROUS_PRECISION = "dexterous_precision"

class GraspPlanner:
    def __init__(self, end_effector_type):
        self.end_effector_type = end_effector_type
        
    def plan_grasp(self, object_mesh, grasp_type=GraspType.PARALLEL):
        """
        Plan an optimal grasp for the given object
        """
        if self.end_effector_type == "parallel":
            return self._plan_parallel_grasp(object_mesh)
        else:
            return self._plan_dexterous_grasp(object_mesh, grasp_type)
    
    def _plan_parallel_grasp(self, object_mesh):
        """
        Plan grasp for parallel gripper
        """
        # Find suitable grasp points by analyzing object geometry
        # This is a simplified approach - real systems use more sophisticated algorithms
        object_points = np.array(object_mesh.vertices)
        
        # Find the object's bounding box
        min_pt = np.min(object_points, axis=0)
        max_pt = np.max(object_points, axis=0)
        
        # Choose grasp point in the middle of the object
        grasp_point = (min_pt + max_pt) / 2
        
        # Choose approach direction based on object orientation
        approach_direction = self._find_optimal_approach_direction(object_mesh)
        
        # Calculate grasp width
        grasp_width = np.linalg.norm(max_pt - min_pt) * 0.6  # 60% of object size
        
        return {
            'position': grasp_point,
            'orientation': approach_direction,
            'grasp_width': grasp_width,
            'grasp_force': 5.0  # Newtons
        }
    
    def _find_optimal_approach_direction(self, object_mesh):
        """
        Determine the best approach direction for grasping
        """
        # Simplified: choose direction of maximum extent
        vertices = np.array(object_mesh.vertices)
        extents = np.max(vertices, axis=0) - np.min(vertices, axis=0)
        major_axis = np.argmax(extents)
        
        approach_direction = np.zeros(3)
        approach_direction[major_axis] = 1.0
        return approach_direction
```

## Grasping Pipeline: Detection -> Approach -> Grasp

The pick-and-place manipulation pipeline consists of three critical phases that must be executed in sequence to achieve successful manipulation.

### Object Detection and Localization

The detection phase identifies target objects in the environment and determines their 6-DOF poses:

```python
class ObjectDetection:
    def __init__(self):
        self.detector = self._initialize_detector()
        
    def detect_objects(self, rgb_image, depth_image):
        """
        Detect and localize objects in the scene
        """
        # Run object detection
        detections = self.detector.detect(rgb_image)
        
        objects = []
        for detection in detections:
            # Use depth information to estimate 3D position
            center_x, center_y = detection['bbox_center']
            depth = depth_image[center_y, center_x]
            
            # Convert to 3D world coordinates
            world_pos = self._depth_to_world(center_x, center_y, depth)
            
            # Estimate object orientation using point cloud analysis
            object_orientation = self._estimate_orientation(
                depth_image, detection['bbox']
            )
            
            objects.append({
                'name': detection['class'],
                'position': world_pos,
                'orientation': object_orientation,
                'bbox': detection['bbox'],
                'confidence': detection['confidence']
            })
        
        return objects
    
    def _depth_to_world(self, x, y, depth):
        """
        Convert depth image coordinates to world coordinates
        """
        # Use camera intrinsic parameters
        fx, fy = self.camera_intrinsics['fx'], self.camera_intrinsics['fy']
        cx, cy = self.camera_intrinsics['cx'], self.camera_intrinsics['cy']
        
        world_x = (x - cx) * depth / fx
        world_y = (y - cy) * depth / fy
        world_z = depth
        
        return np.array([world_x, world_y, world_z])
```

### Approach Phase

The approach phase involves planning and executing a trajectory to position the gripper near the target object:

```python
class ApproachController:
    def __init__(self, robot_interface):
        self.robot = robot_interface
        self.planner = self._initialize_motion_planner()
        
    def approach_object(self, target_object, approach_height=0.1):
        """
        Approach target object with safety considerations
        """
        # Calculate approach position (above target)
        approach_pos = target_object['position'].copy()
        approach_pos[2] += approach_height  # 10cm above object
        
        # Calculate approach orientation
        approach_orient = self._calculate_approach_orientation(target_object)
        
        # Plan trajectory from current position to approach position
        current_pose = self.robot.get_current_pose()
        
        trajectory = self.planner.plan_trajectory(
            start_pose=current_pose,
            target_pose={'position': approach_pos, 'orientation': approach_orient}
        )
        
        # Execute approach trajectory with collision checking
        for waypoint in trajectory:
            if self._is_collision_free(waypoint):
                self.robot.move_to_pose(waypoint)
            else:
                raise Exception("Collision detected during approach")
        
        return True
    
    def _calculate_approach_orientation(self, target_object):
        """
        Calculate optimal approach orientation based on object shape
        """
        # For simple approach: align gripper with object's up direction
        # More complex systems would consider object shape and optimal grasp direction
        object_up = np.array([0, 0, 1])  # Object's up direction in object frame
        world_up = np.array([0, 0, 1])   # World's up direction
        
        # Calculate rotation to align gripper with object
        rotation_matrix = self._align_vectors(world_up, object_up)
        return rotation_matrix
```

### Grasp Execution

The grasp execution phase involves the actual gripping action:

```python
class GraspController:
    def __init__(self, gripper_interface):
        self.gripper = gripper_interface
        self.force_sensor = self._initialize_force_sensor()
        
    def execute_grasp(self, grasp_params, max_force=10.0):
        """
        Execute grasp with force control
        """
        # Move gripper to grasp position
        self.gripper.move_to_width(grasp_params['grasp_width'] + 0.01)  # Open wider initially
        
        # Move to grasp position
        self._position_gripper(grasp_params)
        
        # Close gripper with force control
        success = self._close_gripper_with_force_control(
            target_width=grasp_params['grasp_width'],
            max_force=max_force
        )
        
        if success:
            # Verify grasp success
            grasp_verified = self._verify_grasp()
            if grasp_verified:
                # Lift object slightly
                self._lift_object()
                
        return success and grasp_verified
    
    def _close_gripper_with_force_control(self, target_width, max_force):
        """
        Close gripper while monitoring force to prevent excessive force
        """
        current_width = self.gripper.get_current_width()
        
        while current_width > target_width and self.force_sensor.get_force() < max_force:
            # Gradually close gripper
            current_width -= 0.001  # 1mm per step
            self.gripper.move_to_width(current_width)
            
            # Check for grasp completion
            if self._is_object_grasped():
                break
        
        return self.force_sensor.get_force() <= max_force
    
    def _verify_grasp(self):
        """
        Verify that object is successfully grasped
        """
        # Check force sensor readings
        grip_force = self.force_sensor.get_force()
        if grip_force < 0.5:  # No sufficient grip
            return False
        
        # Check if gripper is holding expected width
        current_width = self.gripper.get_current_width()
        expected_width = self.gripper.get_object_width()
        
        return abs(current_width - expected_width) < 0.005  # 5mm tolerance
```

## Finite State Machine for Grasping Task

The grasping task is well-suited for a finite state machine approach, providing clear state transitions and error handling.

```python
from enum import Enum
import time

class GraspState(Enum):
    IDLE = "idle"
    DETECTING = "detecting"
    PLANNING = "planning"
    APPROACHING = "approaching"
    GRASPING = "grasping"
    VERIFYING = "verifying"
    LIFTING = "lifting"
    PLACING = "placing"
    ERROR = "error"
    COMPLETE = "complete"

class GraspingFSM:
    def __init__(self, robot_controller, perception_module, grasp_controller):
        self.state = GraspState.IDLE
        self.robot = robot_controller
        self.perception = perception_module
        self.grasp_controller = grasp_controller
        
        # Task parameters
        self.target_object = None
        self.destination_pose = None
        self.retry_count = 0
        self.max_retries = 3
        
    def update(self):
        """
        Update state machine based on current state
        """
        if self.state == GraspState.IDLE:
            self._handle_idle()
            
        elif self.state == GraspState.DETECTING:
            self._handle_detection()
            
        elif self.state == GraspState.PLANNING:
            self._handle_planning()
            
        elif self.state == GraspState.APPROACHING:
            self._handle_approach()
            
        elif self.state == GraspState.GRASPING:
            self._handle_grasping()
            
        elif self.state == GraspState.VERIFYING:
            self._handle_verification()
            
        elif self.state == GraspState.LIFTING:
            self._handle_lifting()
            
        elif self.state == GraspState.PLACING:
            self._handle_placing()
            
        elif self.state == GraspState.ERROR:
            self._handle_error()
            
        elif self.state == GraspState.COMPLETE:
            self._handle_complete()
    
    def _handle_idle(self):
        """
        Wait for grasp command
        """
        if self._is_grasp_requested():
            self.target_object = self._get_target_object()
            if self.target_object:
                self.state = GraspState.DETECTING
            else:
                self.state = GraspState.ERROR
    
    def _handle_detection(self):
        """
        Detect and localize target object
        """
        try:
            detected_objects = self.perception.detect_objects()
            
            # Find target object in detected objects
            target = self._find_target_in_detected(self.target_object, detected_objects)
            
            if target:
                self.target_object = target
                self.state = GraspState.PLANNING
            else:
                self.state = GraspState.ERROR
                self._log_error("Target object not found")
                
        except Exception as e:
            self._log_error(f"Detection failed: {e}")
            self.state = GraspState.ERROR
    
    def _handle_planning(self):
        """
        Plan grasp and approach trajectory
        """
        try:
            self.grasp_params = self.grasp_controller.plan_grasp(self.target_object)
            
            if self.grasp_params:
                self.state = GraspState.APPROACHING
            else:
                self.state = GraspState.ERROR
                self._log_error("Grasp planning failed")
                
        except Exception as e:
            self._log_error(f"Grasp planning error: {e}")
            self.state = GraspState.ERROR
    
    def _handle_approach(self):
        """
        Execute approach to object
        """
        try:
            success = self.grasp_controller.approach_object(self.target_object)
            
            if success:
                self.state = GraspState.GRASPING
            else:
                self.state = GraspState.ERROR
                self._log_error("Approach failed")
                
        except Exception as e:
            self._log_error(f"Approach error: {e}")
            self.state = GraspState.ERROR
    
    def _handle_grasping(self):
        """
        Execute grasp action
        """
        try:
            success = self.grasp_controller.execute_grasp(self.grasp_params)
            
            if success:
                self.state = GraspState.VERIFYING
            else:
                if self.retry_count < self.max_retries:
                    self.retry_count += 1
                    self.state = GraspState.PLANNING  # Retry with new grasp
                else:
                    self.state = GraspState.ERROR
                    self._log_error("Grasp failed after retries")
                    
        except Exception as e:
            self._log_error(f"Grasp error: {e}")
            self.state = GraspState.ERROR
    
    def _handle_verification(self):
        """
        Verify grasp success
        """
        try:
            success = self.grasp_controller.verify_grasp()
            
            if success:
                self.state = GraspState.LIFTING
            else:
                if self.retry_count < self.max_retries:
                    self.retry_count += 1
                    self.robot.open_gripper()
                    self.state = GraspState.PLANNING
                else:
                    self.state = GraspState.ERROR
                    self._log_error("Grasp verification failed after retries")
                    
        except Exception as e:
            self._log_error(f"Verification error: {e}")
            self.state = GraspState.ERROR
    
    def _handle_lifting(self):
        """
        Lift object from surface
        """
        try:
            # Move up from object
            current_pose = self.robot.get_current_pose()
            lift_pose = current_pose.copy()
            lift_pose['position'][2] += 0.05  # Lift 5cm
            
            success = self.robot.move_to_pose(lift_pose)
            
            if success:
                self.state = GraspState.PLACING
            else:
                self.state = GraspState.ERROR
                
        except Exception as e:
            self._log_error(f"Lifting error: {e}")
            self.state = GraspState.ERROR
    
    def _handle_placing(self):
        """
        Place object at destination
        """
        try:
            # Move to destination
            success = self.robot.move_to_pose(self.destination_pose)
            
            if success:
                # Release object
                self.robot.open_gripper()
                
                # Move away from object
                self._move_away_from_object()
                self.state = GraspState.COMPLETE
            else:
                self.state = GraspState.ERROR
                
        except Exception as e:
            self._log_error(f"Placing error: {e}")
            self.state = GraspState.ERROR
    
    def _handle_error(self):
        """
        Handle error state
        """
        # Emergency stop and reset
        self.robot.emergency_stop()
        self._reset_system()
        self.state = GraspState.IDLE
    
    def _handle_complete(self):
        """
        Handle completion state
        """
        # Reset for next task
        self.retry_count = 0
        self.state = GraspState.IDLE
        self._log_success("Grasp task completed successfully")
    
    def _log_error(self, message):
        """
        Log error message
        """
        print(f"ERROR: {message}")
        # In a real system, this would log to a file or database
    
    def _log_success(self, message):
        """
        Log success message
        """
        print(f"SUCCESS: {message}")
```

## Complete Grasp Logic Loop Implementation

Here's the complete Python pseudo-code implementation that ties together all components:

```python
#!/usr/bin/env python3
"""
Complete Grasp Control Loop Implementation
"""

import time
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ObjectInfo:
    name: str
    position: list
    orientation: list
    confidence: float


class GraspLogicLoop:
    def __init__(self):
        # Initialize components
        self.fsm = GraspingFSM(
            robot_controller=self._initialize_robot(),
            perception_module=self._initialize_perception(),
            grasp_controller=self._initialize_grasp_controller()
        )
        
        # Control variables
        self.is_running = False
        self.main_thread = None
        self.command_queue = []
        
        # Task parameters
        self.target_object_name = None
        self.destination_pose = None
    
    def start(self):
        """Start the grasp control loop"""
        self.is_running = True
        self.main_thread = threading.Thread(target=self._main_loop)
        self.main_thread.start()
    
    def stop(self):
        """Stop the grasp control loop"""
        self.is_running = False
        if self.main_thread:
            self.main_thread.join()
    
    def request_grasp(self, object_name: str, destination: Dict[str, Any]):
        """Request a grasp operation"""
        self.target_object_name = object_name
        self.destination_pose = destination
        self.fsm.target_object = object_name
        self.fsm.destination_pose = destination
        
        # Transition to detection state if idle
        if self.fsm.state == GraspState.IDLE:
            self.fsm.state = GraspState.DETECTING
    
    def _main_loop(self):
        """Main control loop"""
        while self.is_running:
            try:
                # Update finite state machine
                self.fsm.update()
                
                # Process any commands in queue
                self._process_command_queue()
                
                # Sleep to control loop frequency
                time.sleep(0.01)  # 100Hz update rate
                
            except Exception as e:
                print(f"Grasp control loop error: {e}")
                self.fsm.state = GraspState.ERROR
                time.sleep(0.1)  # Brief pause before continuing
    
    def _process_command_queue(self):
        """Process commands from external sources"""
        # This would handle external commands like emergency stops,
        # new grasp requests, etc.
        pass
    
    def _initialize_robot(self):
        """Initialize robot interface"""
        # In a real implementation, this would connect to the robot
        # via ROS 2 or other communication protocol
        class MockRobot:
            def get_current_pose(self):
                return {'position': [0, 0, 0], 'orientation': [0, 0, 0, 1]}
            
            def move_to_pose(self, pose):
                return True
            
            def open_gripper(self):
                pass
            
            def emergency_stop(self):
                pass
        
        return MockRobot()
    
    def _initialize_perception(self):
        """Initialize perception system"""
        class MockPerception:
            def detect_objects(self):
                # Return mock objects for demonstration
                return [
                    ObjectInfo(
                        name="target_object",
                        position=[0.5, 0.2, 0.0],
                        orientation=[0, 0, 0, 1],
                        confidence=0.9
                    )
                ]
        
        return MockPerception()
    
    def _initialize_grasp_controller(self):
        """Initialize grasp controller"""
        class MockGraspController:
            def plan_grasp(self, object_info):
                return {
                    'position': object_info.position,
                    'orientation': object_info.orientation,
                    'grasp_width': 0.05,
                    'grasp_force': 5.0
                }
            
            def approach_object(self, object_info):
                return True
            
            def execute_grasp(self, grasp_params):
                return True
            
            def verify_grasp(self):
                return True
        
        return MockGraspController()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the grasp system"""
        return {
            'state': self.fsm.state.value,
            'target_object': self.fsm.target_object,
            'current_pose': self.fsm.robot.get_current_pose(),
            'is_running': self.is_running
        }


def main():
    """Example usage of the grasp control system"""
    # Initialize grasp control system
    grasp_system = GraspLogicLoop()
    
    try:
        # Start the system
        grasp_system.start()
        
        # Wait for system to initialize
        time.sleep(1)
        
        # Request a grasp operation
        destination = {
            'position': [0.6, 0.3, 0.2],
            'orientation': [0, 0, 0, 1]
        }
        
        print("Requesting grasp operation...")
        grasp_system.request_grasp("red_cube", destination)
        
        # Monitor progress
        while True:
            status = grasp_system.get_status()
            print(f"Current state: {status['state']}")
            
            if status['state'] in [GraspState.COMPLETE, GraspState.ERROR]:
                break
            
            time.sleep(0.5)
        
        print("Grasp operation completed")
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        grasp_system.stop()


if __name__ == "__main__":
    main()
```

## Summary

This comprehensive control guide has covered the essential components of robotic manipulation and grasping systems. We've explored the fundamental differences between parallel grippers and dexterous hands, including their mechanical and control characteristics. The complete pick-and-place pipeline was detailed with detection, approach, and grasp phases, each with specific control requirements and safety considerations. The finite state machine approach provides a robust framework for managing the complex state transitions required for successful manipulation tasks. Finally, the complete implementation demonstrates how all components integrate in a practical grasp control system. Understanding these concepts is crucial for developing reliable and safe manipulation systems in humanoid robots.