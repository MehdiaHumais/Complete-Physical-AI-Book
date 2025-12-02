# Navigation 2 (Nav2) Stack: Comprehensive Navigation Guide

## Prerequisites

Before diving into this module, students should have:
- Understanding of ROS 2 architecture and concepts
- Knowledge of path planning algorithms (A*, Dijkstra, RRT)
- Experience with costmap representations
- Familiarity with control theory and trajectory generation
- Understanding of behavior trees and finite state machines
- Basic knowledge of differential drive robot kinematics

## Nav2 Stack: Planners, Controllers, and Recoveries

Navigation 2 (Nav2) is the state-of-the-art navigation stack for ROS 2, providing comprehensive path planning, trajectory control, and recovery mechanisms for mobile robots. The system is built around a modular architecture that separates global and local planning, control, and recovery behaviors.

### Global vs Local Planners

**Global Planner**: The global planner computes a complete path from the robot's current position to the goal using a global costmap. It operates on a macro level, considering the overall map to find a feasible route around static obstacles.

The global planner problem formulation:

$$\min_{\tau} \int_0^T C(\tau(t)) dt$$

Subject to:
$$\tau(0) = q_{start}, \tau(T) = q_{goal}$$
$$\tau(t) \in \mathcal{C}_{free}, \forall t \in [0,T]$$

Where $\tau$ is the trajectory, $C$ is the cost function, and $\mathcal{C}_{free}$ is the free configuration space.

Common global planners in Nav2 include:

- **NavFn**: Potential field-based planner using Dijkstra's algorithm
- **Global Planner**: A* implementation with grid-based search
- **Smac Planner**: Sparse-MA* planner for smoother paths in SE(2) or SE(3) space
- **Thunder**: Fast hybrid A* planner optimized for car-like vehicles

```cpp
// Example of global planner cost function
double compute_total_cost(const std::vector<Pose2D>& path, 
                         const Costmap2D& costmap) {
    double total_cost = 0.0;
    
    for (size_t i = 0; i < path.size() - 1; ++i) {
        // Distance cost between consecutive poses
        double distance_cost = euclidean_distance(path[i], path[i+1]);
        
        // Obstacle proximity cost
        double proximity_cost = costmap.getCost(
            path[i].x, path[i].y
        );
        
        // Penalty for high-cost areas
        if (proximity_cost > lethal_cost_) {
            return std::numeric_limits<double>::infinity();
        }
        
        total_cost += distance_cost + 
                     obstacle_weight_ * proximity_cost;
    }
    
    return total_cost;
}
```

**Local Planner**: The local planner generates short-term trajectories to follow the global path while avoiding dynamic obstacles and respecting robot kinodynamics. It operates in the robot's immediate vicinity with higher frequency updates.

Local planning involves the optimization problem:

$$\min_{u(t)} \int_t^{t+T_p} \left[ 
\|x(t) - x_{ref}(t)\|_Q^2 + 
\|u(t)\|_R^2 + 
\alpha \cdot C_{obs}(x(t))
\right] dt$$

Subject to robot dynamics and constraints:

$$\dot{x}(t) = f(x(t), u(t))$$
$$x(t) \in \mathcal{X}_{free}(t)$$
$$u(t) \in \mathcal{U}_{admissible}$$

### DWB (Dynamic Window Approach) Controller

The DWB (Dynamic Window Approach) controller is Nav2's default local planner, implementing a sampling-based approach to generate feasible trajectories in real-time.

The dynamic window represents the set of physically realizable velocities given the robot's current state:

$$\text{DW} = \{(v, \omega) | v_{min} \leq v \leq v_{max}, 
\omega_{min} \leq \omega \leq \omega_{max}\}$$

Where the constraints are derived from:

$$v_{min} = \max(v_{cmd} - a_{max} \cdot \Delta t, v_{min\_allowed})$$
$$v_{max} = \min(v_{cmd} + a_{max} \cdot \Delta t, v_{max\_allowed})$$

The DWB controller evaluates trajectory candidates using a weighted cost function:

$$J(v, \omega) = \alpha \cdot J_{path}(v, \omega) + 
\beta \cdot J_{goal}(v, \omega) + 
\gamma \cdot J_{obst}(v, \omega) + 
\delta \cdot J_{vel}(v, \omega)$$

Where:
- $J_{path}$: Deviation from global path
- $J_{goal}$: Progress toward goal
- $J_{obst}$: Distance to obstacles
- $J_{vel}$: Velocity toward maximum

```cpp
// DWB trajectory evaluation
class DWBController {
public:
    double evaluate_trajectory(
        const geometry_msgs::msg::Twist& cmd_vel,
        const nav_2d_msgs::msg::Path2D& global_plan
    ) {
        // Generate trajectory from command velocity
        auto trajectory = generate_trajectory(cmd_vel);
        
        // Evaluate path following cost
        double path_cost = evaluate_path_cost(trajectory, global_plan);
        
        // Evaluate goal approach cost
        double goal_cost = evaluate_goal_cost(trajectory);
        
        // Evaluate obstacle avoidance cost
        double obstacle_cost = evaluate_obstacle_cost(trajectory);
        
        // Evaluate velocity cost
        double velocity_cost = evaluate_velocity_cost(cmd_vel);
        
        return alpha_ * path_cost + 
               beta_ * goal_cost + 
               gamma_ * obstacle_cost + 
               delta_ * velocity_cost;
    }
    
private:
    double alpha_, beta_, gamma_, delta_;  // Cost weights
    Costmap2D::SharedPtr costmap_ros_;
};
```

### Recovery Behaviors

The recovery system activates when the robot becomes stuck, unable to progress along the planned path. Nav2 implements several recovery behaviors:

**Clear Costmap Recovery**: Clears the local and global costmaps to remove transient obstacles or sensor noise:

```yaml
# Recovery configuration
recovery_plugins: ["spin", "backup", "wait"]
spin:
  plugin: "nav2_recoveries/Spin"
  required_duration: 5.0
  min_duration: 1.0
  max_duration: 10.0
backup:
  plugin: "nav2_recoveries/BackUp"
  backup_dist: -0.15
  backup_speed: 0.025
wait:
  plugin: "nav2_recoveries/Wait"
  sleep_duration: 1.0
```

**Spin Recovery**: Rotates the robot to clear sensor data or relocalize.

**Backup Recovery**: Moves the robot backward to escape local minima.

**Wait Recovery**: Pauses navigation to allow dynamic obstacles to clear.

## Behavior Trees: XML Decision Logic Control

Behavior trees provide a hierarchical, modular approach to robot decision-making, replacing traditional finite state machines with a more flexible and maintainable architecture.

### Behavior Tree Structure

Behavior trees consist of three node types:
- **Action Nodes**: Execute specific behaviors
- **Condition Nodes**: Check boolean conditions
- **Control Nodes**: Manage child node execution flow

### XML Behavior Tree Example

```xml
<root main_tree_to_execute="MainTree">
    <BehaviorTree ID="MainTree">
        <ReactiveSequence>
            <GoalReached>
                <CalculatePath goal="{goal}" path="{path}"/>
                <FollowPath path="{path}" velocity="{velocity}"/>
            </GoalReached>
        </ReactiveSequence>
    </BehaviorTree>
    
    <BehaviorTree ID="NavigateWithReplanning">
        <ReactiveFallback>
            <GoalReached/>
            <ComputePathToPose goal="{goal}" path="{path}"/>
            <Localize/>
            <FollowPath path="{path}"/>
        </ReactiveFallback>
    </BehaviorTree>
</root>
```

### Control Node Types

**Sequence**: Executes children in order until one fails:
```xml
<Sequence>
    <CheckBattery/>     <!-- If fails, sequence stops -->
    <CheckLaser/>       <!-- If fails, sequence stops -->
    <NavigateToGoal/>   <!-- Only executes if above succeed -->
</Sequence>
```

**Fallback (Selector)**: Tries children until one succeeds:
```xml
<Fallback>
    <NavigateToGoal/>   <!-- If succeeds, fallback succeeds -->
    <RecoveryAction/>   <!-- Only tries if navigate fails -->
    <EmergencyStop/>    <!-- Only executes if both fail -->
</Fallback>
```

**ReactiveSequence**: Resets on any child failure:
```xml
<ReactiveSequence>
    <IsGoalReached/>    <!-- If fails, retries from beginning -->
    <PlanToGoal/>       <!-- If fails, retries from beginning -->
    <ExecutePath/>      <!-- If fails, retries from beginning -->
</ReactiveSequence>
```

### Custom Behavior Tree Nodes

```cpp
// Custom BT node implementation
class CheckBatteryNode : public BT::ActionNodeBase
{
public:
    CheckBatteryNode(const std::string& name, 
                     const BT::NodeConfiguration& config)
        : BT::ActionNodeBase(name, config) {}
    
    BT::NodeStatus tick() override {
        // Get battery level from parameter
        double battery_level;
        if (!getInput("battery_level", battery_level)) {
            throw BT::RuntimeError("Missing required input [battery_level]");
        }
        
        // Return SUCCESS if battery level is above threshold
        if (battery_level > 0.2) {  // 20% threshold
            return BT::NodeStatus::SUCCESS;
        }
        
        return BT::NodeStatus::FAILURE;
    }
    
    static BT::PortsList providedPorts() {
        return { BT::InputPort<double>("battery_level") };
    }
};
```

## Costmaps: Inflation, Obstacle, and Static Layers

Costmaps provide the spatial representation of obstacles and navigable areas, combining multiple layers to create a comprehensive cost map used by planners and controllers.

### Costmap Structure

The costmap assigns a cost value to each cell in a grid, where:
- 0 = Free space
- 254 = Lethal obstacle
- 255 = Unknown space

### Static Layer

The static layer represents permanent obstacles from a pre-built map:

$$C_{static}(x,y) = 
\begin{cases} 
254 & \text{if } map(x,y) = occupied \\
0 & \text{if } map(x,y) = free \\
255 & \text{if } map(x,y) = unknown
\end{cases}$$

```yaml
# Static layer configuration
static_layer:
  plugin: "nav2_costmap_2d::StaticLayer"
  map_topic: "map"
  map_subscribe_transient_local: true
  track_unknown_space: true
  use_maximum: false
  unknown_cost_value: 255
  lethal_cost_threshold: 100
  trinary_costmap: true
```

### Obstacle Layer

The obstacle layer processes sensor data to detect dynamic obstacles:

$$C_{obstacle}(x,y) = \max_{sensor}(\text{ray\_trace}(sensor, x, y))$$

The obstacle layer uses ray tracing to propagate sensor readings:

```cpp
void ObstacleLayer::updateBounds(double* min_x, double* min_y, 
                                double* max_x, double* max_y) {
    // Clear previous obstacle data
    std::fill(costmap_.begin(), costmap_.end(), 0);
    
    // Process each obstacle point from sensor
    for (const auto& point : obstacle_points_) {
        unsigned int mx, my;
        if (worldToMap(point.x, point.y, mx, my)) {
            // Mark cell as obstacle
            setCost(mx, my, LETHAL_OBSTACLE);
            
            // Update bounds
            *min_x = std::min(*min_x, point.x);
            *min_y = std::min(*min_y, point.y);
            *max_x = std::max(*max_x, point.x);
            *max_y = std::max(*max_y, point.y);
        }
    }
}
```

### Inflation Layer

The inflation layer expands obstacle costs to create safety margins:

$$C_{inflation}(x,y) = \max_z \left[ 
\text{decay}(distance(x,y,z)) \cdot C_{obstacle}(z) 
\right]$$

Where the decay function typically follows:

$$\text{decay}(d) = \max\left(0, 1 - \frac{d}{inflation\_radius}\right)$$

```yaml
# Inflation layer configuration
inflation_layer:
  plugin: "nav2_costmap_2d::InflationLayer"
  inflation_radius: 0.55
  cost_scaling_factor: 3.0
  inflate_unknown: false
  inflate_around_unknown: false
```

## Tutorial: Configuring Nav2 for Differential Drive Robot

### Step 1: Robot Description and Parameters

First, create a configuration file for your differential drive robot:

```yaml
# config/nav2_params.yaml
amcl:
  ros__parameters:
    use_sim_time: false
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_footprint"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 100.0
    laser_min_range: -1.0
    laser_model_type: "likelihood_field"
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.25
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05

bt_navigator:
  ros__parameters:
    use_sim_time: false
    global_frame: "map"
    robot_base_frame: "base_link"
    odom_topic: "odom"
    default_bt_xml_filename: "navigate_w_replanning_and_recovery.xml"
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_is_stuck_condition_bt_node
    - nav2_goal_reached_condition_bt_node
    - nav2_goal_updated_condition_bt_node
    - nav2_initial_pose_received_condition_bt_node
    - nav2_reinitialize_global_localization_service_bt_node
    - nav2_rate_controller_bt_node
    - nav2_distance_controller_bt_node
    - nav2_speed_controller_bt_node
    - nav2_truncate_path_action_bt_node
    - nav2_goal_updater_node_bt_node
    - nav2_recovery_node_bt_node
    - nav2_pipeline_sequence_bt_node
    - nav2_round_robin_node_bt_node
    - nav2_transform_available_condition_bt_node
    - nav2_time_expired_condition_bt_node
    - nav2_path_expiring_timer_condition
    - nav2_distance_traveled_condition_bt_node
    - nav2_single_trigger_bt_node
    - nav2_is_battery_low_condition_bt_node
    - nav2_navigate_through_poses_action_bt_node
    - nav2_navigate_to_pose_action_bt_node
    - nav2_remove_passed_goals_action_bt_node
    - nav2_planner_selector_bt_node
    - nav2_controller_selector_bt_node
    - nav2_goal_checker_selector_bt_node

controller_server:
  ros__parameters:
    use_sim_time: false
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]
    
    # DWB Controller configuration
    FollowPath:
      plugin: "dwb_core::DWBLocalPlanner"
      debug_trajectory_details: true
      min_vel_x: 0.0
      min_vel_y: 0.0
      max_vel_x: 0.26
      max_vel_y: 0.0
      max_vel_theta: 1.0
      min_speed_xy: 0.0
      max_speed_xy: 0.26
      min_speed_theta: 0.0
      acc_lim_x: 2.5
      acc_lim_y: 0.0
      acc_lim_theta: 3.2
      decel_lim_x: -2.5
      decel_lim_y: 0.0
      decel_lim_theta: -3.2
      velocity_samples: 20
      vx_samples: 20
      vy_samples: 5
      vtheta_samples: 20
      sim_time: 1.7
      linear_granularity: 0.05
      angular_granularity: 0.025
      transform_tolerance: 0.2
      xy_goal_tolerance: 0.25
      trans_stopped_velocity: 0.25
      short_circuit_trajectory_evaluation: true
      stateful: true
      critics: ["RotateToGoal", "Oscillation", "BaseObstacle", "GoalAlign", "PathAlign", "PathDist", "GoalDist"]
      BaseObstacle.scale: 0.02
      PathAlign.scale: 32.0
      PathAlign.forward_point_distance: 0.1
      GoalAlign.scale: 24.0
      GoalAlign.forward_point_distance: 0.1
      PathDist.scale: 32.0
      GoalDist.scale: 24.0
      RotateToGoal.scale: 32.0
      RotateToGoal.slowing_factor: 5.0
      RotateToGoal.lookahead_time: -1.0
```

### Step 2: Costmap Configuration

```yaml
# Continue in nav2_params.yaml
local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: "odom"
      robot_base_frame: "base_link"
      use_sim_time: false
      rolling_window: true
      width: 3
      height: 3
      resolution: 0.05
      origin_x: -1.5
      origin_y: -1.5
      always_send_full_costmap: true
      plugins: ["obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: true
        observation_sources: scan
        scan:
          topic: "/scan"
          max_obstacle_height: 2.0
          clearing: true
          marking: true
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: true
      always_send_full_costmap: true

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: "map"
      robot_base_frame: "base_link"
      use_sim_time: false
      robot_radius: 0.22
      resolution: 0.05
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: true
        observation_sources: scan
        scan:
          topic: "/scan"
          max_obstacle_height: 2.0
          clearing: true
          marking: true
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_topic: "map"
        map_subscribe_transient_local: true
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      always_send_full_costmap: true

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner/NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true
```

### Step 3: Launch Configuration

Create a launch file to bring up the navigation stack:

```xml
<launch>
  <!-- Navigation stack -->
  <node pkg="nav2_lifecycle_manager" exec="lifecycle_manager" name="lifecycle_manager">
    <param name="node_names" value="[map_server, amcl, planner_server, controller_server, bt_navigator]"/>
    <param name="autostart" value="True"/>
  </node>

  <!-- Map server -->
  <node pkg="nav2_map_server" exec="map_server" name="map_server">
    <param name="yaml_filename" value="$(var map_file)"/>
    <param name="topic_name" value="map"/>
    <param name="frame_id" value="map"/>
    <param name="output" value="screen"/>
  </node>

  <!-- AMCL -->
  <node pkg="nav2_amcl" exec="amcl" name="amcl">
    <param name="use_sim_time" value="False"/>
    <param name="install_prefix" value="$(var install_prefix)"/>
  </node>

  <!-- Planner server -->
  <node pkg="nav2_planner" exec="planner_server" name="planner_server" output="screen">
    <param name="use_sim_time" value="False"/>
  </node>

  <!-- Controller server -->
  <node pkg="nav2_controller" exec="controller_server" name="controller_server" output="screen">
    <param name="use_sim_time" value="False"/>
  </node>

  <!-- Behavior tree navigator -->
  <node pkg="nav2_bt_navigator" exec="bt_navigator" name="bt_navigator" output="screen">
    <param name="use_sim_time" value="False"/>
  </node>

  <!-- Recovery server -->
  <node pkg="nav2_recoveries" exec="recoveries_server" name="recoveries_server" output="screen">
    <param name="use_sim_time" value="False"/>
  </node>

  <!-- Velocity smoother -->
  <node pkg="nav2_velocity_smoother" exec="velocity_smoother" name="velocity_smoother" output="screen">
    <param name="use_sim_time" value="False"/>
    <param name="speed_lim_v" value="0.26"/>
    <param name="speed_lim_w" value="1.0"/>
    <param name="accel_lim_v" value="2.5"/>
    <param name="accel_lim_w" value="3.2"/>
  </node>
</launch>
```

## Summary

This comprehensive navigation guide has explored the core components of the Nav2 stack, from the fundamental distinction between global and local planners to the sophisticated behavior tree architecture that controls robot decision-making. The costmap system's layered approach to representing spatial information has been thoroughly explained, showing how static, obstacle, and inflation layers combine to create comprehensive environmental representations.

The detailed tutorial for configuring Nav2 for a differential drive robot provides practical implementation guidance, covering parameter configuration, costmap setup, and launch file creation. Understanding these concepts is essential for developing robust navigation systems that can operate effectively in complex, dynamic environments while maintaining safety and efficiency.