# Physics Guide: Rigid Body Dynamics and Simulation

## Prerequisites

Before diving into this module, students should have:
- Advanced understanding of classical mechanics and Newtonian physics
- Knowledge of vector calculus and differential equations
- Familiarity with linear algebra, especially 3D transformations
- Basic understanding of numerical methods and integration techniques
- Experience with simulation concepts and computational physics
- Understanding of coordinate systems and reference frames

## Rigid Body Dynamics: Newton-Euler Equations of Motion

Rigid body dynamics forms the foundation of physics simulation in robotics, describing how objects move and respond to forces in 3D space. The Newton-Euler equations provide a complete mathematical framework for modeling the translational and rotational motion of rigid bodies.

### Translational Motion: Newton's Second Law

The translational component of rigid body motion is governed by Newton's second law of motion, which states that the rate of change of linear momentum equals the sum of all external forces acting on the body:

$$\frac{d}{dt}(m\mathbf{v}) = \sum \mathbf{F}_{ext}$$

For a constant mass $m$, this simplifies to:

$$m\frac{d\mathbf{v}}{dt} = m\mathbf{a} = \sum \mathbf{F}_{ext}$$

Where:
- $m$ is the mass of the rigid body (kg)
- $\mathbf{v}$ is the linear velocity vector (m/s)
- $\mathbf{a}$ is the linear acceleration vector (m/s²)
- $\sum \mathbf{F}_{ext}$ is the sum of all external forces acting on the body (N)

This equation describes how forces cause linear acceleration, which in turn changes the velocity and position of the rigid body over time.

### Rotational Motion: Euler's Equations

The rotational component is described by Euler's equations, which relate the rate of change of angular momentum to the sum of external torques:

$$\frac{d\mathbf{L}}{dt} = \sum \mathbf{\tau}_{ext}$$

Where $\mathbf{L}$ is the angular momentum vector and $\sum \mathbf{\tau}_{ext}$ is the sum of external torques.

For a rigid body, angular momentum is related to angular velocity through the inertia tensor $\mathbf{I}$:

$$\mathbf{L} = \mathbf{I}\boldsymbol{\omega}$$

Where $\boldsymbol{\omega}$ is the angular velocity vector (rad/s) and $\mathbf{I}$ is the 3x3 inertia tensor (kg·m²).

The inertia tensor is defined as:

$$\mathbf{I} = \begin{bmatrix}
I_{xx} & -I_{xy} & -I_{xz} \\
-I_{yx} & I_{yy} & -I_{yz} \\
-I_{zx} & -I_{zy} & I_{zz}
\end{bmatrix}$$

With elements given by:

$$I_{xx} = \int (y^2 + z^2) dm$$
$$I_{yy} = \int (x^2 + z^2) dm$$
$$I_{zz} = \int (x^2 + y^2) dm$$
$$I_{xy} = I_{yx} = \int xy \, dm$$
$$I_{xz} = I_{zx} = \int xz \, dm$$
$$I_{yz} = I_{zy} = \int yz \, dm$$

### Newton-Euler Equations in Body Frame

When expressed in the body-fixed frame (a coordinate system fixed to the rotating body), the full Newton-Euler equations become:

$$m\mathbf{a} = \sum \mathbf{F}_{ext}$$
$$\mathbf{I}\dot{\boldsymbol{\omega}} + \boldsymbol{\omega} \times (\mathbf{I}\boldsymbol{\omega}) = \sum \boldsymbol{\tau}_{ext}$$

The additional term $\boldsymbol{\omega} \times (\mathbf{I}\boldsymbol{\omega})$ in the rotational equation represents the gyroscopic effect, which occurs when the body rotates about a non-principal axis.

### Integration of Motion Equations

To simulate rigid body motion numerically, these differential equations must be integrated over time. The position and orientation are updated from the velocity and angular velocity:

$$\frac{d\mathbf{r}}{dt} = \mathbf{v}$$
$$\frac{d\mathbf{R}}{dt} = \boldsymbol{\omega} \times \mathbf{R}$$

Where $\mathbf{r}$ is the position vector and $\mathbf{R}$ represents the orientation (typically as a quaternion or rotation matrix).

### Example: Sphere Motion Under Gravity

Consider a sphere of mass $m$ and radius $R$ moving under uniform gravity $\mathbf{g}$:

$$m\mathbf{a} = m\mathbf{g} + \mathbf{F}_{contact}$$
$$\mathbf{I}\dot{\boldsymbol{\omega}} = \boldsymbol{\tau}_{contact}$$

For a solid sphere, the inertia tensor is:
$$\mathbf{I} = \frac{2}{5}mR^2\mathbf{I}_{3x3}$$

Where $\mathbf{I}_{3x3}$ is the 3x3 identity matrix, indicating that the sphere has equal moment of inertia about any axis due to its spherical symmetry.

## Collision Detection: AABB and OBB Algorithms

Collision detection is a fundamental component of physics simulation, determining when and where objects interact. Two primary bounding volume techniques are used: Axis-Aligned Bounding Boxes (AABB) and Oriented Bounding Boxes (OBB).

### Axis-Aligned Bounding Box (AABB) Algorithm

An AABB is a rectangular box aligned with the coordinate axes, defined by its minimum and maximum coordinates. AABBs provide a computationally efficient first-pass collision detection method.

#### AABB Representation

An AABB in 3D space is defined by two 3D points: the minimum corner $\mathbf{min}$ and maximum corner $\mathbf{max}$:

$$\text{AABB} = \{\mathbf{p} \in \mathbb{R}^3 : \mathbf{min}_i \leq \mathbf{p}_i \leq \mathbf{max}_i, \text{ for } i \in \{x,y,z\}\}$$

#### AABB-AABB Collision Detection

Two AABBs collide if and only if they overlap in all three dimensions. For AABB1 and AABB2:

```python
def aabb_collision(aabb1_min, aabb1_max, aabb2_min, aabb2_max):
    """
    Check if two AABBs collide
    
    Args:
        aabb1_min, aabb1_max: (x,y,z) coordinates of first AABB
        aabb2_min, aabb2_max: (x,y,z) coordinates of second AABB
    
    Returns:
        bool: True if collides, False otherwise
    """
    # Check overlap in x dimension
    if aabb1_max[0] < aabb2_min[0] or aabb2_max[0] < aabb1_min[0]:
        return False
    
    # Check overlap in y dimension
    if aabb1_max[1] < aabb2_min[1] or aabb2_max[1] < aabb1_min[1]:
        return False
    
    # Check overlap in z dimension
    if aabb1_max[2] < aabb2_min[2] or aabb2_max[2] < aabb1_min[2]:
        return False
    
    # If we reach here, all dimensions overlap
    return True
```

The computational complexity is O(1) with only 6 comparisons, making AABB tests extremely fast.

#### AABB Construction

To construct an AABB for a complex object, find the minima and maxima of all vertices:

```python
def create_aabb(vertices):
    """
    Create AABB from a list of 3D vertices
    
    Args:
        vertices: List of (x,y,z) tuples or numpy array of shape (n,3)
    
    Returns:
        tuple: (min_point, max_point) representing the AABB
    """
    vertices = np.array(vertices)
    min_point = np.min(vertices, axis=0)
    max_point = np.max(vertices, axis=0)
    return min_point, max_point
```

### Oriented Bounding Box (OBB) Algorithm

An OBB is a rectangular box that can be oriented in any direction, providing tighter fitting than AABB but at increased computational cost.

#### OBB Representation

An OBB is defined by:
- Center point $\mathbf{c}$
- Orientation matrix $\mathbf{R}$ (3x3 rotation matrix)
- Extents (half-sizes) $\mathbf{e} = [e_x, e_y, e_z]$

#### OBB-OBB Collision Detection: Separating Axis Theorem

The Separating Axis Theorem (SAT) states that two convex objects are separate if and only if there exists a plane that separates them. For OBBs, potential separating axes include:

1. The 3 face normals of OBB1
2. The 3 face normals of OBB2  
3. The 9 cross products of each pair of face normals

```python
def obb_collision(obb1_center, obb1_rotation, obb1_extents,
                  obb2_center, obb2_rotation, obb2_extents):
    """
    Check OBB-OBB collision using Separating Axis Theorem
    """
    # Define axes for OBB1 (columns of rotation matrix)
    axes1 = obb1_rotation.T  # Each column is a unit axis
    
    # Define axes for OBB2 (columns of rotation matrix)  
    axes2 = obb2_rotation.T
    
    # Relative translation
    t = obb2_center - obb1_center
    
    # Compute rotation matrix from OBB1 to OBB2 frame
    R = obb2_rotation @ obb1_rotation.T
    
    # Compute translation in OBB2's coordinate frame
    t_obb2 = obb2_rotation.T @ t
    
    # Check separating axes
    for i in range(3):
        for j in range(3):
            # Cross products (potential separating axes)
            axis = np.cross(axes1[i], axes2[j])
            
            # Skip if axis is zero (parallel faces)
            if np.linalg.norm(axis) < 1e-6:
                continue
                
            # Project OBBs onto this axis
            proj1 = np.dot(axes1[i], axes2[j])
            proj2 = np.dot(axes2[j], axes1[i])
            
            # Calculate projected extents
            r1 = obb1_extents[i]
            r2 = obb2_extents[j] * abs(proj1)
            
            # If projections don't overlap, objects are separated
            if abs(np.dot(t, axis)) > r1 + r2:
                return False
    
    # All axes checked - objects collide
    return True
```

The OBB collision test is more complex (O(15) operations) but provides tighter bounds than AABBs, reducing false positives in collision detection.

### Hierarchical Collision Detection

Both AABB and OBB methods are often used in hierarchical structures like bounding volume trees (BVH) to improve performance:

```python
class BoundingVolumeTree:
    def __init__(self):
        self.left = None
        self.right = None
        self.bounding_volume = None  # AABB or OBB
        self.object = None  # Leaf node contains actual object
    
    def intersects(self, other_tree):
        # Fast test: check if bounding volumes intersect
        if not self.bounding_volume.intersects(other_tree.bounding_volume):
            return False
        
        # If both are leaves, check actual objects
        if self.object and other_tree.object:
            return self.object.intersects(other_tree.object)
        
        # Otherwise, recurse down the tree
        if self.left and other_tree.left:
            if self.left.intersects(other_tree.left):
                return True
        if self.right and other_tree.right:
            if self.right.intersects(other_tree.right):
                return True
        
        return False
```

## Physics Engine Comparison: NVIDIA PhysX vs Bullet vs Dart

Modern physics simulation relies on specialized engines that implement efficient collision detection, constraint solving, and integration algorithms. Three leading engines are NVIDIA PhysX, Bullet, and Dart, each with unique strengths.

### NVIDIA PhysX

NVIDIA PhysX is a proprietary physics engine developed by NVIDIA, optimized for GPU acceleration and real-time applications.

**Strengths:**
- **GPU Acceleration**: Extensive CUDA optimization for parallel processing
- **Real-time Performance**: Optimized for interactive applications like games and robotics
- **Professional Support**: Commercial support from NVIDIA
- **Advanced Features**: Cloth simulation, fluid dynamics, destruction systems
- **Multi-platform**: Available on Windows, Linux, macOS, and mobile platforms

**Technical Architecture:**
PhysX employs a parallel execution architecture called "PxFoundation" that allows multi-threading across CPU cores. The solver uses a Projected Gauss-Seidel (PGS) iterative method for constraint solving:

$$\mathbf{J}^T \mathbf{M}^{-1} \mathbf{J} \boldsymbol{\lambda} = \mathbf{b} - \mathbf{J} \mathbf{v}$$

Where:
- $\mathbf{J}$ is the constraint Jacobian matrix
- $\mathbf{M}$ is the mass matrix
- $\boldsymbol{\lambda}$ is the constraint force vector
- $\mathbf{b}$ is the constraint bias vector
- $\mathbf{v}$ is the velocity vector

**Use Case Example:**
```cpp
// PhysX initialization example
PxPhysics* physics = PxCreatePhysics(
    PX_PHYSICS_VERSION, 
    *foundation, 
    PxTolerancesScale(),
    true,
    *physics_insertion_callback
);

// GPU acceleration setup
PxPvd* pvd = PxCreatePvd(*foundation);
PxPvdTransport* transport = PxDefaultPvdSocketTransportCreate("localhost", 5425, 10000);
pvd->connect(*transport, PxPvdInstrumentationFlag::eALL);
```

### Bullet Physics

Bullet is an open-source physics engine that provides industrial-grade features with permissive licensing.

**Strengths:**
- **Open Source**: Free to use with permissive zlib license
- **Research-Friendly**: Excellent documentation and academic support
- **Cross-Platform**: Works on virtually all platforms
- **Multiple Solvers**: Sequential impulse, Dantzig, PGS, and NNK solvers
- **Multi-Paradigm**: Supports both discrete and continuous collision detection

**Technical Architecture:**
Bullet uses a constraint-based approach with a modular design. The Bullet constraint solver can be expressed as:

$$\min_{\boldsymbol{\lambda}} \frac{1}{2} \boldsymbol{\lambda}^T \mathbf{A} \boldsymbol{\lambda} - \boldsymbol{\lambda}^T \mathbf{b}$$

Subject to: $\boldsymbol{\lambda}_{min} \leq \boldsymbol{\lambda} \leq \boldsymbol{\lambda}_{max}$

Where $\mathbf{A} = \mathbf{J} \mathbf{M}^{-1} \mathbf{J}^T$ is the constraint matrix.

**Use Case Example:**
```cpp
// Bullet initialization
btDefaultCollisionConfiguration* collisionConfiguration = 
    new btDefaultCollisionConfiguration();
btCollisionDispatcher* dispatcher = 
    new btCollisionDispatcher(collisionConfiguration);
btBroadphaseInterface* overlappingPairCache = 
    new btDbvtBroadphase();
btSequentialImpulseConstraintSolver* solver = 
    new btSequentialImpulseConstraintSolver;

btDiscreteDynamicsWorld* dynamicsWorld = 
    new btDiscreteDynamicsWorld(
        dispatcher, 
        overlappingPairCache, 
        solver, 
        collisionConfiguration
    );
```

### Dart Physics

Dynamic Animation and Robotics Toolkit (DART) is a specialized physics engine designed for robotics and computer animation applications.

**Strengths:**
- **Robotics-Focused**: Specifically designed for robotics applications
- **Multi-Body Dynamics**: Advanced articulated body algorithms
- **Automatic Differentiation**: Built-in derivatives for optimization
- **Skel-Based Representations**: Skeleton-based articulated body models
- **Advanced Contact Models**: Friction, compliant contact, and soft contacts

**Technical Architecture:**
DART uses a skeleton-based representation for articulated systems. For an articulated body with $n$ degrees of freedom, DART solves:

$$\mathbf{M(q)}\ddot{\mathbf{q}} + \mathbf{C(q, \dot{q})}\dot{\mathbf{q}} + \mathbf{g(q)} = \boldsymbol{\tau} + \mathbf{J}^T \boldsymbol{\lambda}$$

Where:
- $\mathbf{q}$ is the configuration vector
- $\mathbf{M(q)}$ is the mass matrix
- $\mathbf{C(q, \dot{q})}$ contains Coriolis and centrifugal terms
- $\mathbf{g(q)}$ is the gravity vector
- $\boldsymbol{\tau}$ is the joint torque vector
- $\boldsymbol{\lambda}$ is the contact force vector

**Use Case Example:**
```cpp
// DART example
dart::dynamics::SkeletonPtr skeleton = 
    dart::dynamics::Skeleton::create("robot_skeleton");

// Add bodies and joints to form articulated system
dart::dynamics::BodyNodePtr body1 = skeleton->createJointAndBodyNodePair<
    dart::dynamics::RevoluteJoint>(nullptr).second;

dart::dynamics::BodyNodePtr body2 = skeleton->createJointAndBodyNodePair<
    dart::dynamics::RevoluteJoint>(body1).second;

// Physics world simulation
dart::simulation::WorldPtr world = 
    std::make_shared<dart::simulation::World>();
world->addSkeleton(skeleton);
world->step();
```

### Performance Comparison

| Engine | Collision Detection | Constraint Solving | Integration Speed | GPU Support | Robotics Focus |
|--------|-------------------|-------------------|------------------|-------------|----------------|
| PhysX | Fast, optimized | PGS solver | High | Extensive | Low |
| Bullet | Flexible, modular | Multiple solvers | Moderate | Limited | Medium |
| DART | Accurate | Advanced constraints | Moderate | None | High |

## Tutorial: Setting up a Mars Gravity World in Gazebo

Gazebo is a powerful robotics simulation environment that allows for customization of physical parameters, including gravitational acceleration. Setting up a Mars gravity environment enables realistic simulation of robotic missions on the Martian surface.

### Understanding Mars Gravity

Mars has a gravitational acceleration of approximately $3.71 \, \text{m/s}^2$, which is about 38% of Earth's gravity ($9.81 \, \text{m/s}^2$). This reduced gravity significantly affects robotic locomotion, object interactions, and dynamic behavior.

### Step 1: Creating a Custom Gazebo World

Create a new world file `mars_world.world` in your Gazebo worlds directory:

```xml
<?xml version="1.0"?>
<sdf version="1.7">
  <world name="mars_world">
    <!-- Configure Mars gravity -->
    <gravity>0 0 -3.71</gravity>
    
    <!-- Physics engine configuration -->
    <physics name="mars_physics" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <ode>
        <solver>
          <type>quick</type>
          <iters>10</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>0.0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>
    
    <!-- Mars-like environment -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.2 -0.3 -1</direction>
    </light>
    
    <!-- Mars terrain -->
    <model name="mars_terrain">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>1000 1000</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>1000 1000</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.7 0.4 0.2 1</ambient>
            <diffuse>0.7 0.4 0.2 1</diffuse>
            <specular>0.5 0.5 0.5 1</specular>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- Add some obstacles to simulate Mars environment -->
    <model name="rock1">
      <pose>5 0 0.5 0 0 0</pose>
      <link name="link">
        <inertial>
          <mass>5.0</mass>
          <inertia>
            <ixx>0.1</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>0.1</iyy>
            <iyz>0.0</iyz>
            <izz>0.1</izz>
          </inertia>
        </inertial>
        <collision name="collision">
          <geometry>
            <sphere>
              <radius>0.3</radius>
            </sphere>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <sphere>
              <radius>0.3</radius>
            </sphere>
          </geometry>
          <material>
            <ambient>0.2 0.2 0.2 1</ambient>
            <diffuse>0.3 0.3 0.3 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- Example robot model -->
    <model name="mars_rover">
      <pose>0 0 1.0 0 0 0</pose>
      <link name="chassis">
        <inertial>
          <mass>20.0</mass>
          <inertia>
            <ixx>2.0</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>3.0</iyy>
            <iyz>0.0</iyz>
            <izz>4.0</izz>
          </inertia>
        </inertial>
        <collision name="collision">
          <geometry>
            <box>
              <size>1.0 0.8 0.4</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1.0 0.8 0.4</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

### Step 2: Launching the Mars World

To launch the Mars gravity world in Gazebo, use the following command:

```bash
gazebo ~/gazebo_worlds/mars_world.world
```

Alternatively, create a launch file for ROS 2 integration:

```xml
<?xml version="1.0"?>
<launch>
  <arg name="world" default="~/gazebo_worlds/mars_world.world"/>
  
  <node name="gazebo" pkg="gazebo_ros" exec="gazebo" args="-v 4 -s libgazebo_ros_factory.so -s libgazebo_ros_state.so $(var world)"/>
</launch>
```

### Step 3: Validating Mars Gravity Effects

Create a simple test to verify the gravity is correctly set:

```python
#!/usr/bin/env python3
"""
Test script to validate Mars gravity in Gazebo simulation
"""
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
import time

class MarsGravityValidator(Node):
    def __init__(self):
        super().__init__('mars_gravity_validator')
        
        # Create spawn service client
        self.spawn_cli = self.create_client(SpawnEntity, '/spawn_entity')
        while not self.spawn_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Spawn service not available, waiting again...')
        
        # Timer to periodically check object position
        self.timer = self.create_timer(0.1, self.check_fall_speed)
        self.drop_object_created = False
        self.initial_time = None
        self.initial_height = 10.0  # Drop from 10m height
        
    def spawn_drop_object(self):
        """Spawn an object to test gravity"""
        req = SpawnEntity.Request()
        req.name = "gravity_test_sphere"
        req.xml = """
        <sdf version="1.6">
          <model name="gravity_test_sphere">
            <pose>0 0 10 0 0 0</pose>
            <link name="link">
              <inertial>
                <mass>1.0</mass>
                <inertia>
                  <ixx>0.01</ixx>
                  <ixy>0.0</ixy>
                  <ixz>0.0</ixz>
                  <iyy>0.01</iyy>
                  <iyz>0.0</iyz>
                  <izz>0.01</izz>
                </inertia>
              </inertial>
              <collision name="collision">
                <geometry>
                  <sphere>
                    <radius>0.1</radius>
                  </sphere>
                </geometry>
              </collision>
              <visual name="visual">
                <geometry>
                  <sphere>
                    <radius>0.1</radius>
                  </sphere>
                </geometry>
              </visual>
            </link>
          </model>
        </sdf>"""
        req.initial_pose.position.x = 0.0
        req.initial_pose.position.y = 0.0
        req.initial_pose.position.z = self.initial_height
        req.initial_pose.orientation.w = 1.0
        
        future = self.spawn_cli.call_async(req)
        future.add_done_callback(self.spawn_callback)
        
    def spawn_callback(self, future):
        """Handle spawn response"""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info('Drop object spawned successfully')
                self.drop_object_created = True
                self.initial_time = self.get_clock().now().nanoseconds / 1e9
            else:
                self.get_logger().error(f'Failed to spawn object: {response.status_message}')
        except Exception as e:
            self.get_logger().error(f'Spawn service call failed: {e}')
    
    def check_fall_speed(self):
        """Check the falling speed to validate gravity"""
        if not self.drop_object_created:
            if self.get_clock().now().nanoseconds / 1e9 > 1.0:  # Wait 1 second before spawning
                self.spawn_drop_object()
            return
        
        # In a real implementation, you would get the object's position
        # from Gazebo topics or services to calculate velocity
        current_time = self.get_clock().now().nanoseconds / 1e9
        time_elapsed = current_time - self.initial_time
        
        # Calculate expected position under Mars gravity: h = h0 - 0.5 * g * t^2
        expected_height = self.initial_height - 0.5 * 3.71 * time_elapsed**2
        
        self.get_logger().info(f'Time: {time_elapsed:.2f}s, Expected height: {expected_height:.2f}m')
        
        # If object hits ground (height ~ 0), verify it matches Mars gravity prediction
        if expected_height <= 0.1:  # Allow small margin
            theoretical_time = (2 * self.initial_height / 3.71)**0.5
            self.get_logger().info(f'Object should hit ground in {theoretical_time:.2f}s')
            self.get_logger().info(f'Actual time: {time_elapsed:.2f}s')
            self.timer.cancel()  # Stop monitoring

def main(args=None):
    rclpy.init(args=args)
    node = MarsGravityValidator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 4: Adjusting Robot Controllers for Mars Gravity

When operating robots in Mars gravity, controller gains may need adjustment. For a simple PD controller:

```python
class MarsAdaptiveController:
    def __init__(self, earth_g=9.81, mars_g=3.71):
        self.earth_g = earth_g
        self.mars_g = mars_g
        self.gravity_ratio = self.mars_g / self.earth_g
        
        # Adjust controller gains based on gravity ratio
        # Reduce gains proportionally to gravity reduction
        self.kp = 100.0 * self.gravity_ratio  # Proportional gain
        self.kd = 10.0 * self.gravity_ratio  # Derivative gain
    
    def compute_torque(self, error, error_derivative):
        """Compute control torque with Mars-adapted gains"""
        torque = self.kp * error + self.kd * error_derivative
        return torque
```

### Physics Validation Tests

To validate that your Mars gravity world is correctly configured, perform these tests:

1. **Free Fall Test**: Drop an object and measure its acceleration
2. **Pendulum Test**: Test pendulum period (should be $\sqrt{g_{earth}/g_{mars}}$ times longer)
3. **Projectile Motion Test**: Verify trajectories match reduced gravity
4. **Stability Test**: Ensure static objects remain stable

The reduced gravity on Mars affects many aspects of robot operation:
- Locomotion becomes easier but requires different control strategies
- Manipulation tasks may require less force but have different dynamics
- Balance control algorithms need adjustment for the different weight distribution
- Energy consumption patterns change due to reduced gravitational forces

## Summary

This comprehensive physics guide has covered the essential mathematics of rigid body dynamics through the Newton-Euler equations, providing the theoretical foundation for understanding how objects move and interact in 3D space. The collision detection algorithms (AABB and OBB) with their respective advantages and implementation details have been thoroughly explained, forming the basis for efficient physics simulation.

The comparison between NVIDIA PhysX, Bullet, and Dart physics engines highlighted their unique strengths and appropriate use cases for different robotics applications. Finally, the practical tutorial on setting up a Mars gravity world in Gazebo demonstrated how to customize physics parameters for specific mission requirements.

Understanding these physics fundamentals is crucial for developing realistic and accurate robotic simulations that can bridge the gap between virtual testing and real-world deployment.