# Mathematical Guide to Robot Description and Transformations

## Prerequisites

Before diving into this module, students should have:
- Understanding of linear algebra (vectors, matrices, transformations)
- Basic knowledge of Python programming
- Familiarity with XML syntax
- Understanding of coordinate systems and reference frames
- Knowledge of rigid body mechanics and kinematics
- Basic understanding of ROS 2 concepts

## URDF (Unified Robot Description Format) Fundamentals

The Unified Robot Description Format (URDF) is an XML-based format used to describe robots in ROS. URDF provides a complete description of robot components including their physical properties, visual appearance, and kinematic relationships. Understanding URDF is crucial for robot simulation, visualization, and kinematic analysis.

### Core URDF XML Tags

#### `<link>` Element

The `<link>` element represents a rigid body in the robot structure. Each link has specific properties that define its physical and visual characteristics:

```xml
<link name="link_name">
  <!-- Visual properties define how the link appears -->
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="1 1 1"/>
    </geometry>
    <material name="red">
      <color rgba="1 0 0 1"/>
    </material>
  </visual>
  
  <!-- Collision properties define how the link interacts physically -->
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="1 1 1"/>
    </geometry>
  </collision>
  
  <!-- Inertial properties define the link's physical dynamics -->
  <inertial>
    <mass value="1.0"/>
    <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
  </inertial>
</link>
```

The `<link>` element contains three essential child elements:

**Visual Element**: Defines the appearance of the link in visualization tools. It includes:
- Origin: Position and orientation offset from the link's frame
- Geometry: Shape definition (box, cylinder, sphere, or mesh)
- Material: Color and visual properties

**Collision Element**: Defines the collision geometry used in physics simulation. It specifies:
- Shape for collision detection
- Offset from the link's reference frame
- The collision geometry may differ from visual geometry for computational efficiency

**Inertial Element**: Defines the physical properties needed for dynamics simulation:
- Mass: The mass of the link
- Inertia tensor: 6 independent values describing mass distribution

#### `<joint>` Element

The `<joint>` element defines the connection between two links, specifying allowed motions:

```xml
<joint name="joint_name" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0.5 0 0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="3.0"/>
</joint>
```

Key attributes of joints include:
- **Type**: Specifies the degrees of freedom (revolute, continuous, prismatic, fixed, floating, planar)
- **Parent/Child**: Defines the connection between links
- **Origin**: Position and orientation of the joint relative to the parent
- **Axis**: Direction of motion for revolute and prismatic joints
- **Limit**: For revolute joints, specifies angle limits; for prismatic joints, position limits

#### Joint Types

1. **Revolute**: 1 DOF rotation with limits
2. **Continuous**: 1 DOF rotation without limits
3. **Prismatic**: 1 DOF translation with limits
4. **Fixed**: No DOF (welded connection)
5. **Floating**: 6 DOF rigid body motion
6. **Planar**: 3 DOF planar motion

#### `<inertial>` Element and Collision Meshes

The `<inertial>` element is crucial for realistic physics simulation:

```xml
<inertial>
  <origin xyz="0.1 0.0 0.0" rpy="0 0 0"/>
  <mass value="2.0"/>
  <inertia ixx="0.4" ixy="0.01" ixz="0.02" iyy="0.3" iyz="0.01" izz="0.2"/>
</inertial>
```

The inertia tensor represents how mass is distributed in the link:

$$I = \begin{bmatrix}
I_{xx} & -I_{xy} & -I_{xz} \\
-I_{yx} & I_{yy} & -I_{yz} \\
-I_{zx} & -I_{zy} & I_{zz}
\end{bmatrix}$$

Where each element represents:
- $I_{xx} = \int (y^2 + z^2) dm$ (moment of inertia about x-axis)
- $I_{yy} = \int (x^2 + z^2) dm$ (moment of inertia about y-axis)
- $I_{zz} = \int (x^2 + y^2) dm$ (moment of inertia about z-axis)
- $I_{xy} = I_{yx} = \int xy \, dm$ (product of inertia)
- $I_{xz} = I_{zx} = \int xz \, dm$ (product of inertia)
- $I_{yz} = I_{zy} = \int yz \, dm$ (product of inertia)

### Collision Meshes

For complex geometries, URDF supports mesh-based collision detection:

```xml
<collision>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <mesh filename="package://package_name/meshes/complex_part.stl"/>
  </geometry>
</collision>
```

Collision meshes can be simplified versions of visual meshes to optimize performance while maintaining accuracy for collision detection.

## Complete URDF for a 3-DOF Robot Leg

Here's a comprehensive URDF description for a 3-DOF robot leg with hip, knee, and ankle joints:

```xml
<?xml version="1.0"?>
<robot name="robot_leg_3dof">
  <!-- Material definitions -->
  <material name="black">
    <color rgba="0.1 0.1 0.1 1.0"/>
  </material>
  <material name="red">
    <color rgba="0.8 0.2 0.2 1.0"/>
  </material>
  <material name="blue">
    <color rgba="0.2 0.2 0.8 1.0"/>
  </material>
  <material name="green">
    <color rgba="0.2 0.8 0.2 1.0"/>
  </material>

  <!-- Base link - connects to robot torso -->
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <mass value="0.5"/>
      <inertia ixx="0.0005" ixy="0.0" ixz="0.0" 
               iyy="0.0005" iyz="0.0" izz="0.00025"/>
    </inertial>
  </link>

  <!-- Hip joint - rotation around Y axis -->
  <joint name="hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="thigh_link"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="200" velocity="3.0"/>
    <dynamics damping="1.0" friction="0.1"/>
  </joint>

  <!-- Thigh link -->
  <link name="thigh_link">
    <visual>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.4" radius="0.04"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.4" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.04" ixy="0.0" ixz="0.0" 
               iyy="0.02" iyz="0.0" izz="0.04"/>
    </inertial>
  </link>

  <!-- Knee joint - rotation around Y axis -->
  <joint name="knee_joint" type="revolute">
    <parent link="thigh_link"/>
    <child link="shank_link"/>
    <origin xyz="0 0 -0.5" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.785" upper="2.356" effort="200" velocity="3.0"/>
    <dynamics damping="1.0" friction="0.1"/>
  </joint>

  <!-- Shank (lower leg) link -->
  <link name="shank_link">
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.35" radius="0.035"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.35" radius="0.035"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <mass value="1.5"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" 
               iyy="0.01" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <!-- Ankle joint - rotation around Y axis -->
  <joint name="ankle_joint" type="revolute">
    <parent link="shank_link"/>
    <child link="foot_link"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="150" velocity="2.5"/>
    <dynamics damping="0.8" friction="0.1"/>
  </joint>

  <!-- Foot link -->
  <link name="foot_link">
    <visual>
      <origin xyz="0 0 -0.025" rpy="0 0 0"/>
      <geometry>
        <box size="0.25 0.15 0.05"/>
      </geometry>
      <material name="green"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.025" rpy="0 0 0"/>
      <geometry>
        <box size="0.25 0.15 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 -0.025" rpy="0 0 0"/>
      <mass value="0.8"/>
      <inertia ixx="0.0025" ixy="0.0" ixz="0.0" 
               iyy="0.0045" iyz="0.0" izz="0.0065"/>
    </inertial>
  </link>
</robot>
```

This URDF describes a 3-DOF leg with:
- Hip joint: Rotational movement around Y-axis
- Knee joint: Rotational movement around Y-axis
- Ankle joint: Rotational movement around Y-axis
- Appropriate physical properties for dynamics simulation
- Visual and collision geometry for both simulation and visualization

## Homogeneous Transformation Matrices in TF2

The Transform Library 2 (TF2) in ROS 2 uses homogeneous transformation matrices to represent coordinate frame transformations. Homogeneous coordinates allow translation, rotation, and scaling to be combined into a single matrix operation.

### Mathematical Foundation

A 3D point $(x, y, z)$ in Cartesian coordinates is represented in homogeneous coordinates as a 4-element vector:

$$\mathbf{P}_h = \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}$$

The additional "1" in the fourth component enables translation to be represented as a matrix multiplication, making all affine transformations (rotation, translation, scaling) expressible as matrix operations.

### 4x4 Transformation Matrix

A general 4x4 homogeneous transformation matrix has the form:

$$T = \begin{bmatrix}
R_{3 \times 3} & \mathbf{d} \\
\mathbf{0}^T & 1
\end{bmatrix} = 
\begin{bmatrix}
r_{11} & r_{12} & r_{13} & d_x \\
r_{21} & r_{22} & r_{23} & d_y \\
r_{31} & r_{32} & r_{33} & d_z \\
0 & 0 & 0 & 1
\end{bmatrix}$$

Where:
- $R_{3 \times 3}$ is the 3x3 rotation matrix
- $\mathbf{d} = [d_x, d_y, d_z]^T$ is the 3x1 translation vector
- $\mathbf{0}^T$ is the 1x3 zero vector

### Rotation Submatrix

The rotation submatrix $R$ describes the orientation of one frame relative to another:

$$R = \begin{bmatrix}
\mathbf{x}_B \cdot \mathbf{x}_A & \mathbf{x}_B \cdot \mathbf{y}_A & \mathbf{x}_B \cdot \mathbf{z}_A \\
\mathbf{y}_B \cdot \mathbf{x}_A & \mathbf{y}_B \cdot \mathbf{y}_A & \mathbf{y}_B \cdot \mathbf{z}_A \\
\mathbf{z}_B \cdot \mathbf{x}_A & \mathbf{z}_B \cdot \mathbf{y}_A & \mathbf{z}_B \cdot \mathbf{z}_A
\end{bmatrix}$$

Where $\mathbf{x}_A, \mathbf{y}_A, \mathbf{z}_A$ are the unit vectors of frame A and $\mathbf{x}_B, \mathbf{y}_B, \mathbf{z}_B$ are the unit vectors of frame B.

### Basic Transformation Matrices

**Translation Matrix** (moving by $[t_x, t_y, t_z]$):
$$T_{trans} = \begin{bmatrix}
1 & 0 & 0 & t_x \\
0 & 1 & 0 & t_y \\
0 & 0 & 1 & t_z \\
0 & 0 & 0 & 1
\end{bmatrix}$$

**Rotation about X-axis** by angle $\theta$:
$$R_x(\theta) = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & \cos\theta & -\sin\theta & 0 \\
0 & \sin\theta & \cos\theta & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}$$

**Rotation about Y-axis** by angle $\theta$:
$$R_y(\theta) = \begin{bmatrix}
\cos\theta & 0 & \sin\theta & 0 \\
0 & 1 & 0 & 0 \\
-\sin\theta & 0 & \cos\theta & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}$$

**Rotation about Z-axis** by angle $\theta$:
$$R_z(\theta) = \begin{bmatrix}
\cos\theta & -\sin\theta & 0 & 0 \\
\sin\theta & \cos\theta & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}$$

### Composing Transformations

Multiple transformations can be combined by matrix multiplication. The order of multiplication is crucial, as matrix multiplication is not commutative:

$$T_{final} = T_n \cdot T_{n-1} \cdot \ldots \cdot T_2 \cdot T_1$$

### Transformation Application

To transform a point from frame A to frame B using transformation $T_{A}^{B}$:

$$\mathbf{P}_B = T_{A}^{B} \cdot \mathbf{P}_A$$

The inverse transformation from frame B to frame A is:

$$T_{B}^{A} = (T_{A}^{B})^{-1}$$

For transformation matrices, the inverse has a special form:

$$(T_{A}^{B})^{-1} = \begin{bmatrix}
(R_{A}^{B})^T & -(R_{A}^{B})^T \mathbf{d}_{A}^{B} \\
\mathbf{0}^T & 1
\end{bmatrix}$$

### Python Implementation in TF2

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

def create_homogeneous_transform(translation, rotation_matrix):
    """
    Create a 4x4 homogeneous transformation matrix
    
    Args:
        translation: 3-element array [x, y, z] for translation
        rotation_matrix: 3x3 rotation matrix
    
    Returns:
        4x4 homogeneous transformation matrix
    """
    T = np.eye(4)  # Start with 4x4 identity matrix
    T[0:3, 0:3] = rotation_matrix  # Place rotation in upper-left 3x3
    T[0:3, 3] = translation        # Place translation in last column
    return T

def transform_point(T, point):
    """
    Transform a 3D point using homogeneous transformation
    
    Args:
        T: 4x4 homogeneous transformation matrix
        point: 3-element array [x, y, z]
    
    Returns:
        Transformed 3-element array [x', y', z']
    """
    # Convert to homogeneous coordinates
    point_h = np.array([point[0], point[1], point[2], 1.0])
    
    # Apply transformation
    transformed_h = T @ point_h
    
    # Convert back to Cartesian coordinates
    return transformed_h[0:3]

def create_rotation_from_axis_angle(axis, angle):
    """
    Create rotation matrix from axis-angle representation
    
    Args:
        axis: 3-element unit vector representing rotation axis
        angle: Rotation angle in radians
    
    Returns:
        3x3 rotation matrix
    """
    # Normalize the axis vector
    axis = axis / np.linalg.norm(axis)
    
    # Calculate rotation matrix using axis-angle formula
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    one_minus_cos = 1 - cos_theta
    
    x, y, z = axis
    
    R = np.array([
        [cos_theta + x*x*one_minus_cos,     x*y*one_minus_cos - z*sin_theta, x*z*one_minus_cos + y*sin_theta],
        [y*x*one_minus_cos + z*sin_theta,  cos_theta + y*y*one_minus_cos,    y*z*one_minus_cos - x*sin_theta],
        [z*x*one_minus_cos - y*sin_theta,  z*y*one_minus_cos + x*sin_theta,  cos_theta + z*z*one_minus_cos]
    ])
    
    return R

# Example usage
translation = np.array([1.0, 2.0, 3.0])  # Translation vector
rotation_matrix = create_rotation_from_axis_angle(np.array([0, 0, 1]), np.pi/4)  # 45-degree rotation about Z

T = create_homogeneous_transform(translation, rotation_matrix)
point_A = np.array([1.0, 0.0, 0.0])
point_B = transform_point(T, point_A)

print(f"Original point: {point_A}")
print(f"Transformed point: {point_B}")
```

## Quaternions: Avoiding Gimbal Lock

Quaternions provide a robust mathematical representation for 3D rotations that avoids the gimbal lock problem inherent in Euler angle representations.

### The Gimbal Lock Problem

Gimbal lock occurs when two of the three rotational axes become aligned, resulting in the loss of one degree of freedom. This happens with Euler angle representations when the pitch angle reaches ±90° (π/2 radians).

In Euler angles (roll, pitch, yaw), the rotation sequence typically follows a pattern like ZYX, where rotations are applied in a specific order. When pitch reaches ±90°, the roll and yaw axes become collinear, meaning changes in roll and yaw produce the same physical rotation.

### Mathematical Representation

A quaternion is a 4-element vector $(x, y, z, w)$ that represents a rotation in 3D space:

$$\mathbf{q} = [x, y, z, w] = \begin{bmatrix} x \\ y \\ z \\ w \end{bmatrix}$$

Where:
- $(x, y, z)$ represents the rotation axis scaled by $\sin(\theta/2)$
- $w$ represents $\cos(\theta/2)$
- $\theta$ is the rotation angle

For a rotation of angle $\theta$ about unit axis $\mathbf{u} = [u_x, u_y, u_z]$:

$$\mathbf{q} = \begin{bmatrix}
u_x \sin(\theta/2) \\
u_y \sin(\theta/2) \\
u_z \sin(\theta/2) \\
\cos(\theta/2)
\end{bmatrix}$$

### Why Quaternions Avoid Gimbal Lock

Quaternions represent rotations in a 4-dimensional space, which inherently avoids the topological issues that cause gimbal lock in 3D representations like Euler angles. The mathematical structure of quaternions ensures that:

1. **No Singular Points**: Unlike Euler angles, quaternions don't have problematic orientations where degrees of freedom are lost
2. **Double Covering**: Each rotation has exactly two quaternion representations ($\mathbf{q}$ and $-\mathbf{q}$), providing global smoothness
3. **Compact Representation**: Only 4 parameters compared to 9 for rotation matrices

### Converting Between Representations

**Quaternion to Rotation Matrix**:

$$R = \begin{bmatrix}
1 - 2(y^2 + z^2) & 2(xy - wz) & 2(xz + wy) \\
2(xy + wz) & 1 - 2(x^2 + z^2) & 2(yz - wx) \\
2(xz - wy) & 2(yz + wx) & 1 - 2(x^2 + y^2)
\end{bmatrix}$$

**Euler Angles to Quaternion** (ZYX sequence):

For Euler angles $(\phi, \theta, \psi)$ representing rotations about Z, Y, X axes respectively:

$$\mathbf{q} = \begin{bmatrix}
\sin\frac{\phi}{2}\cos\frac{\theta}{2}\cos\frac{\psi}{2} - \cos\frac{\phi}{2}\sin\frac{\theta}{2}\sin\frac{\psi}{2} \\
\cos\frac{\phi}{2}\sin\frac{\theta}{2}\cos\frac{\psi}{2} + \sin\frac{\phi}{2}\cos\frac{\theta}{2}\sin\frac{\psi}{2} \\
\cos\frac{\phi}{2}\cos\frac{\theta}{2}\sin\frac{\psi}{2} - \sin\frac{\phi}{2}\sin\frac{\theta}{2}\cos\frac{\psi}{2} \\
\cos\frac{\phi}{2}\cos\frac{\theta}{2}\cos\frac{\psi}{2} + \sin\frac{\phi}{2}\sin\frac{\theta}{2}\sin\frac{\psi}{2}
\end{bmatrix}$$

### Python Implementation with Quaternions

```python
import numpy as np

class Quaternion:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        """
        Initialize quaternion with (x, y, z, w) components
        """
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.normalize()
    
    def normalize(self):
        """
        Normalize the quaternion to unit length
        """
        norm = np.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)
        if norm > 0:
            self.x /= norm
            self.y /= norm
            self.z /= norm
            self.w /= norm
    
    def to_rotation_matrix(self):
        """
        Convert quaternion to 3x3 rotation matrix
        """
        x, y, z, w = self.x, self.y, self.z, self.w
        
        # Calculate rotation matrix elements
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        
        return R
    
    def from_axis_angle(self, axis, angle):
        """
        Create quaternion from axis-angle representation
        """
        axis = np.array(axis) / np.linalg.norm(axis)  # Normalize axis
        sin_half_angle = np.sin(angle / 2)
        
        self.x = axis[0] * sin_half_angle
        self.y = axis[1] * sin_half_angle
        self.z = axis[2] * sin_half_angle
        self.w = np.cos(angle / 2)
        
        self.normalize()
    
    def multiply(self, other):
        """
        Multiply this quaternion by another quaternion
        """
        # Hamilton product: q1 * q2
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        
        return Quaternion(x, y, z, w)
    
    def rotate_vector(self, vector):
        """
        Rotate a 3D vector using this quaternion
        """
        # Convert vector to pure quaternion (0, vector)
        q_vector = Quaternion(vector[0], vector[1], vector[2], 0)
        q_conjugate = Quaternion(-self.x, -self.y, -self.z, self.w)
        
        # Rotate: v' = q * v * q_conjugate
        temp = self.multiply(q_vector)
        rotated = temp.multiply(q_conjugate)
        
        return np.array([rotated.x, rotated.y, rotated.z])

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion
    Using ZYX rotation order
    """
    # Convert to half angles
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    
    # Calculate quaternion components
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return Quaternion(x, y, z, w)

# Example usage
# Create a quaternion representing 45-degree rotation about Z-axis
q1 = Quaternion()
q1.from_axis_angle([0, 0, 1], np.pi/4)

# Create another quaternion representing 30-degree rotation about X-axis
q2 = Quaternion()
q2.from_axis_angle([1, 0, 0], np.pi/6)

# Combine rotations
combined_rotation = q1.multiply(q2)

# Rotate a vector
original_vector = np.array([1, 0, 0])
rotated_vector = combined_rotation.rotate_vector(original_vector)

print(f"Original vector: {original_vector}")
print(f"Rotated vector: {rotated_vector}")
print(f"Rotation matrix:\n{combined_rotation.to_rotation_matrix()}")
```

### TF2 Quaternion Implementation

In ROS 2 and TF2, quaternions are commonly represented in geometry_msgs with the (x, y, z, w) convention:

```python
from geometry_msgs.msg import Quaternion
import tf2_ros
from tf2_ros import TransformStamped
import math

def create_quaternion_from_euler(roll, pitch, yaw):
    """
    Create quaternion message from Euler angles
    """
    # Precompute half angles
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    
    return q

def main():
    # Example: Create a transform with quaternion rotation
    transform = TransformStamped()
    
    # Set translation (position)
    transform.transform.translation.x = 1.0
    transform.transform.translation.y = 2.0
    transform.transform.translation.z = 3.0
    
    # Set rotation (quaternion from Euler angles)
    euler_angles = (0.0, 0.0, math.pi/4)  # 45-degree rotation about Z
    quaternion = create_quaternion_from_euler(*euler_angles)
    transform.transform.rotation = quaternion
    
    # The transform now represents a translation with rotation
    print(f"Translation: ({transform.transform.translation.x}, "
          f"{transform.transform.translation.y}, {transform.transform.translation.z})")
    print(f"Rotation quaternion: ({transform.transform.rotation.x}, "
          f"{transform.transform.rotation.y}, {transform.transform.rotation.z}, "
          f"{transform.transform.rotation.w})")

if __name__ == '__main__':
    main()
```

## Summary

This mathematical guide has provided a comprehensive overview of robot description and transformation mathematics essential for ROS 2 applications. We've explored the URDF format with its core elements (`<link>`, `<joint>`, `<inertial>`), showing how to create complete robot descriptions including collision meshes.

The complete 3-DOF robot leg URDF demonstrates practical application of these concepts with realistic physical properties and appropriate joint types. The mathematical foundation of homogeneous transformation matrices in TF2 has been thoroughly explained, including their structure, composition, and implementation.

Finally, we've addressed the crucial topic of quaternions and their advantages over Euler angles, particularly in avoiding gimbal lock. The mathematical representations and practical implementations shown provide the foundation for robust 3D rotation handling in robotic applications.

Understanding these mathematical concepts is essential for creating accurate robot simulations, implementing precise control algorithms, and developing reliable navigation systems in robotics applications.