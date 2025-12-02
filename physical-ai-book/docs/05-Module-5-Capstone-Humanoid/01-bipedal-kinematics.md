# Bipedal Kinematics: Physics & Mathematics of Humanoid Locomotion

## Prerequisites

Before diving into this module, students should have:
- Advanced understanding of linear algebra, vector calculus, and matrix operations
- Knowledge of classical mechanics and rigid body dynamics
- Familiarity with coordinate systems and transformation matrices
- Understanding of control theory and system dynamics
- Experience with robotics kinematics and inverse kinematics solvers
- Mathematical proficiency in solving systems of equations and optimization

## Kinematics: Inverse vs Forward Kinematics

The kinematic analysis of bipedal robots involves two fundamental approaches: Forward Kinematics (FK) and Inverse Kinematics (IK). These complementary methods form the mathematical foundation for robotic motion planning and control.

### Forward Kinematics (FK)

Forward kinematics computes the end-effector position and orientation given the joint angles. For a humanoid leg with $n$ joints, the transformation from the base to the end-effector is expressed as:

$$\mathbf{T}_{n}^{0} = \prod_{i=1}^{n} \mathbf{A}_{i}(\theta_i) = \mathbf{A}_1(\theta_1) \mathbf{A}_2(\theta_2) \cdots \mathbf{A}_n(\theta_n)$$

Where $\mathbf{A}_i(\theta_i)$ is the homogeneous transformation matrix for joint $i$:

$$\mathbf{A}_i = \begin{bmatrix}
\mathbf{R}_i & \mathbf{p}_i \\
\mathbf{0}^T & 1
\end{bmatrix}$$

For a revolute joint in the Denavit-Hartenberg convention:

$$\mathbf{A}_i = \begin{bmatrix}
\cos\theta_i & -\sin\theta_i\cos\alpha_i & \sin\theta_i\sin\alpha_i & a_i\cos\theta_i \\
\sin\theta_i & \cos\theta_i\cos\alpha_i & -\cos\theta_i\sin\alpha_i & a_i\sin\theta_i \\
0 & \sin\alpha_i & \cos\alpha_i & d_i \\
0 & 0 & 0 & 1
\end{bmatrix}$$

Where:
- $\theta_i$ is the joint angle
- $d_i$ is the link offset
- $a_i$ is the link length
- $\alpha_i$ is the link twist

### Inverse Kinematics (IK)

Inverse kinematics solves the opposite problem: given a desired end-effector pose, find the required joint angles. This is typically formulated as an optimization problem:

$$\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} \| \mathbf{f}(\boldsymbol{\theta}) - \mathbf{x}_{desired} \|_2^2$$

Subject to joint limit constraints:
$$\boldsymbol{\theta}_{min} \leq \boldsymbol{\theta} \leq \boldsymbol{\theta}_{max}$$

### Mathematical Solutions

**Analytical Solution**: For simple kinematic chains (e.g., 6-DOF manipulator), closed-form solutions exist:

$$\theta_1 = \text{atan2}(y, x) \pm \text{acos}\left(\frac{r^2 + l_1^2 - l_2^2}{2rl_1}\right)$$

Where $r = \sqrt{x^2 + y^2}$ for a 2D planar manipulator.

**Iterative Solution**: For complex chains, numerical methods like the Jacobian-based approach:

$$\Delta\boldsymbol{\theta} = \mathbf{J}^{\dagger} \Delta\mathbf{x}$$

Where $\mathbf{J}^{\dagger}$ is the pseudoinverse of the Jacobian matrix.

### Jacobian Matrix

The Jacobian relates joint velocities to end-effector velocities:

$$\dot{\mathbf{x}} = \mathbf{J}(\boldsymbol{\theta}) \dot{\boldsymbol{\theta}}$$

Where the Jacobian is:

$$\mathbf{J} = \begin{bmatrix}
\frac{\partial f_1}{\partial \theta_1} & \frac{\partial f_1}{\partial \theta_2} & \cdots & \frac{\partial f_1}{\partial \theta_n} \\
\frac{\partial f_2}{\partial \theta_1} & \frac{\partial f_2}{\partial \theta_2} & \cdots & \frac{\partial f_2}{\partial \theta_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial \theta_1} & \frac{\partial f_m}{\partial \theta_2} & \cdots & \frac{\partial f_m}{\partial \theta_n}
\end{bmatrix}$$

### Implementation Example

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

def forward_kinematics(joint_angles, link_lengths):
    """
    Compute FK for a simple planar 3-DOF leg
    """
    theta1, theta2, theta3 = joint_angles
    
    # Link transformations
    T1 = np.array([[np.cos(theta1), -np.sin(theta1), 0, 0],
                   [np.sin(theta1), np.cos(theta1), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    
    T2 = np.array([[np.cos(theta2), -np.sin(theta2), 0, link_lengths[0]],
                   [np.sin(theta2), np.cos(theta2), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    
    T3 = np.array([[np.cos(theta3), -np.sin(theta3), 0, link_lengths[1]],
                   [np.sin(theta3), np.cos(theta3), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    
    # Combined transformation
    T_total = T1 @ T2 @ T3
    end_effector_pos = T_total[:3, 3]
    
    return end_effector_pos

def jacobian_2d(joint_angles, link_lengths):
    """
    Compute 2D Jacobian for planar manipulator
    """
    theta1, theta2, theta3 = joint_angles
    l1, l2, l3 = link_lengths
    
    # Jacobian elements for 2D planar case
    J = np.zeros((2, 3))
    
    # Partial derivatives of end-effector position w.r.t. joint angles
    J[0, 0] = -l1*np.sin(theta1) - l2*np.sin(theta1 + theta2) - l3*np.sin(theta1 + theta2 + theta3)  # dx/dtheta1
    J[0, 1] = -l2*np.sin(theta1 + theta2) - l3*np.sin(theta1 + theta2 + theta3)  # dx/dtheta2
    J[0, 2] = -l3*np.sin(theta1 + theta2 + theta3)  # dx/dtheta3
    
    J[1, 0] = l1*np.cos(theta1) + l2*np.cos(theta1 + theta2) + l3*np.cos(theta1 + theta2 + theta3)   # dy/dtheta1
    J[1, 1] = l2*np.cos(theta1 + theta2) + l3*np.cos(theta1 + theta2 + theta3)   # dy/dtheta2
    J[1, 2] = l3*np.cos(theta1 + theta2 + theta3)   # dy/dtheta3
    
    return J

def inverse_kinematics_2d(target_pos, link_lengths, initial_guess=[0, 0, 0], max_iter=100):
    """
    Solve 2D inverse kinematics using Jacobian pseudo-inverse method
    """
    joint_angles = np.array(initial_guess)
    tolerance = 1e-6
    
    for i in range(max_iter):
        current_pos = forward_kinematics(joint_angles, link_lengths)[:2]
        error = target_pos - current_pos
        
        if np.linalg.norm(error) < tolerance:
            break
            
        J = jacobian_2d(joint_angles, link_lengths)
        J_pseudo = np.linalg.pinv(J)
        
        delta_theta = J_pseudo @ error
        joint_angles += delta_theta
    
    return joint_angles
```

## Balance: Mathematical Condition for Stability

The stability of bipedal robots fundamentally depends on maintaining the center of mass (CoM) within the support polygon defined by the contact points with the ground.

### Support Polygon Definition

For a bipedal robot, the support polygon is the convex hull of ground contact points. For double support (both feet on ground):

$$\text{Support Polygon} = \text{conv}(\mathbf{p}_1, \mathbf{p}_2, \ldots, \mathbf{p}_n)$$

Where $\mathbf{p}_i$ are the contact points.

### CoM Stability Condition

The mathematical condition for static stability is:

$$\mathbf{p}_{CoM} \in \text{Support Polygon}$$

Where $\mathbf{p}_{CoM} = (x_{CoM}, y_{CoM})$ is the projection of the center of mass onto the ground plane.

### Dynamic Stability

For dynamic walking, the stability condition becomes more complex. The instantaneous capture point must lie within the support polygon:

$$\mathbf{p}_{capture} = \mathbf{p}_{CoM} + \frac{\mathbf{v}_{CoM}}{\omega} \in \text{Support Polygon}$$

Where $\omega = \sqrt{\frac{g}{z_{CoM}}}$ and $z_{CoM}$ is the height of the center of mass above the ground.

### ZMP-Based Stability

The Zero Moment Point (ZMP) must lie within the support polygon for dynamic stability:

$$\mathbf{p}_{ZMP} = (x_{ZMP}, y_{ZMP}) \in \text{Support Polygon}$$

### Implementation for Stability Analysis

```python
def is_stable(com_pos, support_polygon):
    """
    Check if CoM is within support polygon
    """
    def point_in_polygon(point, polygon):
        """Ray casting algorithm to check if point is inside polygon"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n+1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    return point_in_polygon((com_pos[0], com_pos[1]), support_polygon)
```

## Zero Moment Point (ZMP) Theory Derivation

The Zero Moment Point is a critical concept in bipedal locomotion that determines where the ground reaction force would need to act to produce zero moment about that point.

### Force and Moment Analysis

Consider a human body in contact with the ground. The total external forces and moments about the center of mass are:

$$\sum \mathbf{F}_{ext} = m\ddot{\mathbf{r}}_{CoM} = m\mathbf{g} + \sum_i \mathbf{F}_i$$

$$\sum \mathbf{M}_{ext} = I\ddot{\boldsymbol{\theta}} = \sum_i (\mathbf{r}_i - \mathbf{r}_{CoM}) \times \mathbf{F}_i$$

Where:
- $m$ is the total mass
- $\mathbf{r}_{CoM}$ is the center of mass position
- $\mathbf{F}_i$ are the contact forces at point $\mathbf{r}_i$
- $I$ is the moment of inertia

### ZMP Definition

The ZMP is the point on the ground where the moment of the ground reaction forces is zero. Taking moments about an arbitrary point $\mathbf{p}$ on the ground plane:

$$\mathbf{M}_p = \sum_i (\mathbf{r}_i - \mathbf{p}) \times \mathbf{F}_i = \mathbf{0}$$

For the ZMP point $\mathbf{p}_{ZMP}$, we specifically consider the point where the moment about the horizontal axes is zero:

$$\mathbf{M}_{ZMP} = \sum_i (\mathbf{r}_i - \mathbf{p}_{ZMP}) \times \mathbf{F}_i = \mathbf{0}$$

### ZMP Derivation

Starting from the moment equation about the center of mass:

$$\sum_i (\mathbf{r}_i - \mathbf{r}_{CoM}) \times \mathbf{F}_i = I\ddot{\boldsymbol{\theta}}$$

For the ZMP point $\mathbf{p}_{ZMP}$:

$$\sum_i (\mathbf{p}_{ZMP} - \mathbf{r}_i) \times \mathbf{F}_i = \mathbf{0}$$

Expanding this equation:

$$\sum_i \mathbf{p}_{ZMP} \times \mathbf{F}_i - \sum_i \mathbf{r}_i \times \mathbf{F}_i = \mathbf{0}$$

$$\mathbf{p}_{ZMP} \times \sum_i \mathbf{F}_i = \sum_i \mathbf{r}_i \times \mathbf{F}_i$$

Using the force equation $\sum_i \mathbf{F}_i = m\mathbf{g} + m\ddot{\mathbf{r}}_{CoM}$:

$$\mathbf{p}_{ZMP} \times (m\mathbf{g} + m\ddot{\mathbf{r}}_{CoM}) = \sum_i \mathbf{r}_i \times \mathbf{F}_i$$

### 2D Simplification

For 2D planar motion where the robot only moves in the sagittal plane (x-z plane), we consider moments about the y-axis:

$$x_{ZMP} = x_{CoM} - \frac{z_{CoM} \cdot \ddot{x}_{CoM}}{g + \ddot{z}_{CoM}}$$

$$y_{ZMP} = y_{CoM} - \frac{z_{CoM} \cdot \ddot{y}_{CoM}}{g + \ddot{z}_{CoM}}$$

### ZMP Stability Criterion

For dynamic stability, the ZMP trajectory must lie within the convex hull of the feet contact points throughout the gait cycle:

$$x_{ZMP}(t) \in [x_{min}(t), x_{max}(t)]$$
$$y_{ZMP}(t) \in [y_{min}(t), y_{max}(t)]$$

Where $[x_{min}(t), x_{max}(t)]$ and $[y_{min}(t), y_{max}(t)]$ define the support polygon boundaries at time $t$.

### ZMP Control Implementation

```python
def compute_zmp(com_pos, com_vel, com_acc, gravity=9.81):
    """
    Compute ZMP from CoM state
    """
    x_com, y_com, z_com = com_pos
    x_vel, y_vel, z_vel = com_vel
    x_acc, y_acc, z_acc = com_acc
    
    # ZMP equations for 2D case
    x_zmp = x_com - (z_com * x_acc) / (gravity + z_acc)
    y_zmp = y_com - (z_com * y_acc) / (gravity + z_acc)
    
    return np.array([x_zmp, y_zmp, 0.0])

def generate_zmp_trajectory(step_length, step_width, step_height, gait_params):
    """
    Generate ZMP trajectory for walking
    """
    # Simplified ZMP trajectory generation
    # This would be more complex in practice with smooth transitions
    
    t = np.linspace(0, 1, 100)  # Normalize gait cycle
    
    # Double support phase ZMP trajectory (simplified)
    zmp_x = step_length/2 * np.sin(np.pi * t)  # Simplified sinusoidal model
    zmp_y = step_width/2 * np.ones_like(t)    # Simplified constant lateral position
    
    return zmp_x, zmp_y
```

## Humanoid Leg Kinematic Model (6 DOF)

[Mermaid Chart: Simplified 6-DOF Humanoid Leg Kinematic Diagram showing: 
1. Hip joint with 3 DOF (flexion/extension, abduction/adduction, internal/external rotation)
2. Knee joint with 1 DOF (flexion/extension) 
3. Ankle joint with 2 DOF (dorsi/plantar flexion, inversion/eversion)
4. Links labeled: Hip link, Thigh, Shank, Foot
5. Coordinate frames at each joint with Z-axes aligned with rotation axes
6. Joint angles labeled: θ1 (hip flexion), θ2 (hip abduction), θ3 (hip rotation), θ4 (knee flexion), θ5 (ankle flexion), θ6 (ankle abduction)
7. End-effector frame at the center of the foot sole
8. Mathematical notation showing the kinematic chain: Base → Hip → Thigh → Knee → Shank → Ankle → Foot]

### 6-DOF Leg Structure

A typical humanoid leg consists of 6 degrees of freedom distributed across three joints:

**Hip Joint (3 DOF)**:
- $\theta_1$: Flexion/Extension (sagittal plane)
- $\theta_2$: Abduction/Adduction (coronal plane)  
- $\theta_3$: Internal/External Rotation (transverse plane)

**Knee Joint (1 DOF)**:
- $\theta_4$: Flexion/Extension (sagittal plane)

**Ankle Joint (2 DOF)**:
- $\theta_5$: Dorsi/Plantar Flexion (sagittal plane)
- $\theta_6$: Inversion/Eversion (coronal plane)

### Denavit-Hartenberg Parameters

| Joint | $a_i$ (link length) | $\alpha_i$ (link twist) | $d_i$ (link offset) | $\theta_i$ (joint angle) |
|-------|-------------------|--------------------|------------------|-----------------------|
| 1 (Hip X) | 0 | -90° | $d_1$ | $\theta_1$ |
| 2 (Hip Y) | $a_2$ | 0 | 0 | $\theta_2$ |
| 3 (Hip Z) | $a_3$ | 90° | 0 | $\theta_3$ |
| 4 (Knee) | $a_4$ | 0 | 0 | $\theta_4$ |
| 5 (Ankle X) | $a_5$ | 0 | 0 | $\theta_5$ |
| 6 (Ankle Y) | $a_6$ | 90° | $d_6$ | $\theta_6$ |

### Forward Kinematics for 6-DOF Leg

The complete transformation matrix for the 6-DOF leg:

$$\mathbf{T}_{foot}^{base} = \prod_{i=1}^{6} \mathbf{A}_i(\theta_i)$$

Where each $\mathbf{A}_i$ is computed using the DH parameters and represents the transformation from frame $i-1$ to frame $i$.

### Jacobian for 6-DOF Leg

The complete Jacobian matrices for the 6-DOF leg system include both linear and angular components:

$$\mathbf{J}_{linear} = \begin{bmatrix}
\mathbf{z}_1 \times (\mathbf{p}_{ee} - \mathbf{p}_1) & \cdots & \mathbf{z}_6 \times (\mathbf{p}_{ee} - \mathbf{p}_6)
\end{bmatrix}$$

$$\mathbf{J}_{angular} = \begin{bmatrix}
\mathbf{z}_1 & \mathbf{z}_2 & \cdots & \mathbf{z}_6
\end{bmatrix}$$

Where $\mathbf{z}_i$ is the axis of rotation for joint $i$ in the end-effector frame.

### Implementation Example for 6-DOF Leg

```python
class Humanoid6DOFLeg:
    def __init__(self, link_lengths):
        self.a1, self.a2, self.a3, self.a4, self.a5, self.a6 = link_lengths
        self.d1, self.d6 = 0.05, 0.1  # Hip and ankle offsets
    
    def forward_kinematics(self, joint_angles):
        """
        Compute FK for 6-DOF humanoid leg
        joint_angles: [theta1, theta2, theta3, theta4, theta5, theta6]
        """
        t1, t2, t3, t4, t5, t6 = joint_angles
        
        # Compute individual transformation matrices
        T1 = self._dh_transform(t1, 0, self.d1, -90*np.pi/180)
        T2 = self._dh_transform(t2, self.a2, 0, 0)
        T3 = self._dh_transform(t3, self.a3, 0, 90*np.pi/180)
        T4 = self._dh_transform(t4, self.a4, 0, 0)
        T5 = self._dh_transform(t5, self.a5, 0, 0)
        T6 = self._dh_transform(t6, self.a6, self.d6, 90*np.pi/180)
        
        # Combined transformation
        T_total = T1 @ T2 @ T3 @ T4 @ T5 @ T6
        
        return T_total
    
    def _dh_transform(self, theta, a, d, alpha):
        """DH transformation matrix"""
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])
    
    def compute_jacobian(self, joint_angles):
        """
        Compute Jacobian for 6-DOF leg
        """
        # Get current end-effector position
        T_ee = self.forward_kinematics(joint_angles)
        p_ee = T_ee[:3, 3]
        
        # Compute joint positions and z-axes
        joint_positions = []
        z_axes = []
        
        current_T = np.eye(4)
        
        for i, theta in enumerate(joint_angles):
            # Compute transformation to current joint
            if i == 0:
                T_joint = self._dh_transform(theta, 0, self.d1, -90*np.pi/180)
            elif i == 1:
                T_joint = self._dh_transform(theta, self.a2, 0, 0)
            elif i == 2:
                T_joint = self._dh_transform(theta, self.a3, 0, 90*np.pi/180)
            elif i == 3:
                T_joint = self._dh_transform(theta, self.a4, 0, 0)
            elif i == 4:
                T_joint = self._dh_transform(theta, self.a5, 0, 0)
            elif i == 5:
                T_joint = self._dh_transform(theta, self.a6, self.d6, 90*np.pi/180)
            
            current_T = current_T @ T_joint
            joint_positions.append(current_T[:3, 3])
            
            # z-axis of current joint in world frame
            z_axis = current_T[:3, 2]
            z_axes.append(z_axis)
        
        # Compute Jacobian
        J = np.zeros((6, 6))  # [linear; angular]
        
        for i in range(6):
            # Linear velocity part
            if i == 0:
                J[:3, i] = np.cross(z_axes[i], (p_ee - joint_positions[i]))
            else:
                J[:3, i] = np.cross(z_axes[i], (p_ee - joint_positions[i]))
            
            # Angular velocity part
            J[3:, i] = z_axes[i]
        
        return J
```

## Summary

This comprehensive physics and mathematics guide has explored the fundamental kinematic principles underlying bipedal locomotion in humanoid robots. We've examined the complementary approaches of forward and inverse kinematics, demonstrating both analytical and numerical solutions for motion planning. The balance stability conditions, particularly the requirement for the center of mass to remain within the support polygon, were thoroughly analyzed with both static and dynamic considerations. The Zero Moment Point theory was derived mathematically, providing the foundation for stable walking pattern generation. Finally, the 6-DOF humanoid leg model was detailed with appropriate kinematic representations and control implementations. Understanding these mathematical foundations is essential for developing stable and efficient bipedal robots capable of complex locomotion tasks.