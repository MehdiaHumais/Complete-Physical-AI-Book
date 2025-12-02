# Sensor Simulation Guide

## Prerequisites

Before diving into this module, students should have:
- Understanding of 3D geometry and ray-triangle intersection mathematics
- Knowledge of sensor physics and noise modeling
- Basic understanding of coordinate systems and transformations
- Familiarity with XML and SDF syntax for robot descriptions
- Experience with ROS 2 concepts and message types
- Understanding of Gaussian distributions and noise modeling

## LiDAR Sensor Simulation: Ray Casting Theory

LiDAR (Light Detection and Ranging) sensors are crucial for robotics applications, providing precise 3D mapping and obstacle detection. In simulation, LiDAR sensors are implemented using ray casting algorithms that mimic the physical behavior of laser beams.

### Ray Casting Fundamentals

Ray casting is a rendering technique where rays are cast from a viewpoint into a scene to determine what objects are visible. In LiDAR simulation, each ray represents a laser beam that travels until it intersects with an object, returning the distance to that object.

The mathematical representation of a ray in 3D space is:

$$\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$$

Where:
- $\mathbf{o}$ is the ray origin (sensor position)
- $\mathbf{d}$ is the ray direction (unit vector)
- $t$ is the parameter along the ray (distance from origin)
- $\mathbf{r}(t)$ is a point on the ray

For LiDAR simulation, multiple rays are cast simultaneously, each representing a different angular direction in the sensor's field of view.

### LiDAR Sensor Parameters

A LiDAR sensor is characterized by several parameters:

- **Vertical Field of View (FOV)**: Range of elevation angles
- **Horizontal Field of View**: Range of azimuth angles  
- **Angular Resolution**: Minimum angle between adjacent rays
- **Range**: Minimum and maximum detectable distances
- **Scan Rate**: Number of full scans per second
- **Rays per Scan**: Number of rays in each horizontal scan

### Velodyne VLP-16 Simulation

The Velodyne VLP-16 is a popular 16-channel LiDAR sensor with the following specifications:
- 16 laser channels arranged vertically
- 360° horizontal field of view
- -15° to +15° vertical field of view
- 0.1° horizontal angular resolution
- 0.2° vertical angular resolution
- Range: 0.2m to 100m
- 10Hz rotation rate

In Gazebo simulation, the VLP-16 is implemented using ray tracing where each of the 16 channels corresponds to a different elevation angle.

### Ray-Surface Intersection

For LiDAR simulation, the core algorithm determines the intersection between each ray and the geometric surfaces in the environment. For a triangle with vertices $\mathbf{v_1}$, $\mathbf{v_2}$, and $\mathbf{v_3}$, the intersection with ray $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ is computed using the Möller-Trumbore algorithm:

1. Calculate the triangle's normal: $\mathbf{n} = (\mathbf{v_2} - \mathbf{v_1}) \times (\mathbf{v_3} - \mathbf{v_1})$
2. Calculate ray-triangle intersection parameter: $t = \frac{(\mathbf{v_1} - \mathbf{o}) \cdot \mathbf{n}}{\mathbf{d} \cdot \mathbf{n}}$
3. Verify the intersection point lies within the triangle using barycentric coordinates

### LiDAR Noise Modeling

Real LiDAR sensors exhibit various noise characteristics including:

- **Range Noise**: Small variations in measured distances
- **Angular Noise**: Slight variations in beam direction
- **Intensity Noise**: Variations in returned signal strength

Range noise can be modeled as Gaussian noise with zero mean and standard deviation proportional to the measured distance:

$$\text{measured\_range} = \text{true\_range} + \mathcal{N}(0, \sigma \cdot \text{true\_range})$$

Where $\sigma$ is the range-dependent noise factor.

## IMU Sensor Simulation: Noise Models

Inertial Measurement Units (IMUs) combine accelerometers and gyroscopes to measure motion and orientation. Simulation of IMUs must include realistic noise models to accurately represent sensor behavior.

### Accelerometer Noise Model

Accelerometers measure linear acceleration and are subject to several types of noise:

**Gaussian White Noise**: The primary noise source in accelerometers, modeled as:

$$\mathbf{a}_{measured} = \mathbf{a}_{true} + \boldsymbol{\eta}_a$$

Where $\boldsymbol{\eta}_a \sim \mathcal{N}(\mathbf{0}, \sigma_a^2\mathbf{I})$ represents the Gaussian noise vector.

**Bias Drift**: Long-term drift in the zero reading, often modeled as a random walk:

$$\mathbf{b}_a(t) = \mathbf{b}_a(t_0) + \int_{t_0}^{t} \boldsymbol{\eta}_{bias}(\tau) d\tau$$

Where $\boldsymbol{\eta}_{bias} \sim \mathcal{N}(\mathbf{0}, \sigma_{bias}^2\mathbf{I})$.

**Scale Factor Errors**: Deviations from the ideal measurement scale, typically constant but can drift over time.

### Gyroscope Noise Model

Gyroscopes measure angular velocity with their own noise characteristics:

**Gaussian White Noise**: Similar to accelerometers:

$$\boldsymbol{\omega}_{measured} = \boldsymbol{\omega}_{true} + \boldsymbol{\eta}_g$$

**Bias Drift**: Also modeled as a random walk:

$$\mathbf{b}_g(t) = \mathbf{b}_g(t_0) + \int_{t_0}^{t} \boldsymbol{\eta}_{g,bias}(\tau) d\tau$$

**Rate Random Walk**: A type of noise that accumulates over time, particularly important for gyros.

### IMU Simulation Implementation

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

class IMUSensor:
    def __init__(self, dt=0.01):
        self.dt = dt  # Time step
        self.g = 9.81  # Gravitational acceleration
        
        # Noise parameters (typical for MEMS IMU)
        self.accel_noise_std = 0.017  # m/s^2
        self.gyro_noise_std = 0.0012  # rad/s
        self.accel_bias_std = 1e-4    # m/s^2/sqrt(Hz)
        self.gyro_bias_std = 1e-5     # rad/s/sqrt(Hz)
        
        # Initialize biases
        self.accel_bias = np.random.normal(0, self.accel_bias_std * np.sqrt(dt), 3)
        self.gyro_bias = np.random.normal(0, self.gyro_bias_std * np.sqrt(dt), 3)
    
    def simulate(self, true_accel, true_gyro, orientation):
        """
        Simulate IMU measurements with noise
        
        Args:
            true_accel: True linear acceleration in world frame (3,)
            true_gyro: True angular velocity in body frame (3,)
            orientation: Current orientation as rotation matrix (3,3)
        
        Returns:
            accel_measured: Measured acceleration (3,)
            gyro_measured: Measured angular velocity (3,)
        """
        # Transform true acceleration to body frame
        true_accel_body = orientation.T @ (true_accel + np.array([0, 0, -self.g]))
        
        # Add Gaussian white noise
        accel_noise = np.random.normal(0, self.accel_noise_std, 3)
        gyro_noise = np.random.normal(0, self.gyro_noise_std, 3)
        
        # Add noise and bias
        accel_measured = true_accel_body + accel_noise + self.accel_bias
        gyro_measured = true_gyro + gyro_noise + self.gyro_bias
        
        # Update bias (random walk)
        self.accel_bias += np.random.normal(0, self.accel_bias_std * np.sqrt(self.dt), 3)
        self.gyro_bias += np.random.normal(0, self.gyro_bias_std * np.sqrt(self.dt), 3)
        
        return accel_measured, gyro_measured
```

## Camera Sensor Simulation

Camera sensors provide visual information crucial for perception tasks. In simulation, cameras can be configured to output RGB, depth, and optical flow information.

### RGB-D Camera Simulation

RGB-D cameras provide both color (RGB) and depth information. The depth information is crucial for 3D reconstruction and object detection tasks.

The depth sensor in Gazebo uses ray casting similar to LiDAR but for each pixel in the image. For a pinhole camera model, a ray is cast from the optical center through each pixel coordinate:

$$\mathbf{r} = \mathbf{C} + s \cdot \mathbf{d}$$

Where:
- $\mathbf{C}$ is the camera center
- $\mathbf{d}$ is the ray direction in world coordinates
- $s$ is the distance parameter

The depth value is the distance to the closest intersection point with scene geometry.

### Camera Intrinsic Parameters

Camera intrinsic parameters relate 3D world points to 2D image coordinates:

$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = 
\begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
\begin{bmatrix} x \\ y \\ z \end{bmatrix}$$

Where:
- $(f_x, f_y)$ are focal lengths in pixels
- $(c_x, c_y)$ are principal point coordinates
- $(u, v)$ are image coordinates
- $(x, y, z)$ are normalized 3D coordinates

### Optical Flow Simulation

Optical flow represents the apparent motion of objects, surfaces, and edges in a visual scene. In simulation, optical flow can be computed from consecutive depth images using motion field calculations:

$$\mathbf{v} = \frac{d\mathbf{x}}{dt}$$

Where $\mathbf{v}$ is the optical flow vector and $\mathbf{x}$ represents pixel coordinates.

## LiDAR SDF Implementation

The following SDF snippet demonstrates how to add a LiDAR sensor to a robot link in Gazebo:

```xml
<sdf version="1.6">
  <model name="robot_with_lidar">
    <link name="base_link">
      <pose>0 0 0.1 0 0 0</pose>
      <collision name="collision">
        <geometry>
          <box>
            <size>0.5 0.3 0.2</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>0.5 0.3 0.2</size>
          </box>
        </geometry>
      </visual>
    </link>
    
    <!-- Velodyne VLP-16 LiDAR sensor -->
    <link name="lidar_link">
      <pose>0 0 0.3 0 0 0</pose> <!-- 0.3m above base -->
      <collision name="lidar_collision">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.1</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="lidar_visual">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.1</length>
          </cylinder>
        </geometry>
      </visual>
      
      <!-- LiDAR sensor specification -->
      <sensor name="vlp16_sensor" type="ray">
        <always_on>1</always_on>
        <update_rate>10</update_rate>
        <pose>0 0 0 0 0 0</pose>
        
        <ray>
          <!-- Number of rays in horizontal and vertical directions -->
          <scan>
            <horizontal>
              <samples>1872</samples> <!-- 0.1° resolution over 360° -->
              <resolution>1</resolution>
              <min_angle>-3.14159</min_angle> <!-- -π radians -->
              <max_angle>3.14159</max_angle>  <!-- π radians -->
            </horizontal>
            <vertical>
              <samples>16</samples> <!-- 16 channels -->
              <resolution>1</resolution>
              <min_angle>-0.2618</min_angle> <!-- -15° in radians -->
              <max_angle>0.2618</max_angle>  <!-- +15° in radians -->
            </vertical>
          </scan>
          
          <!-- Range parameters -->
          <range>
            <min>0.2</min> <!-- Minimum range: 0.2m -->
            <max>100.0</max> <!-- Maximum range: 100m -->
            <resolution>0.01</resolution> <!-- Range resolution: 1cm -->
          </range>
          
          <!-- Noise model for the sensor -->
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.01</stddev> <!-- 1cm standard deviation -->
          </noise>
        </ray>
        
        <!-- Plugin to publish sensor data -->
        <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
          <ros>
            <namespace>/robot</namespace>
            <remapping>~/out:=scan</remapping>
          </ros>
          <output_type>sensor_msgs/LaserScan</output_type>
        </plugin>
      </sensor>
    </link>
    
    <!-- Connect the LiDAR to the base -->
    <joint name="lidar_joint" type="fixed">
      <parent>base_link</parent>
      <child>lidar_link</child>
    </joint>
  </model>
</sdf>
```

This SDF specification creates a robot model with a Velodyne VLP-16 LiDAR sensor mounted on top. The sensor configuration includes:

- **1872 horizontal samples** to achieve 0.1° angular resolution over 360°
- **16 vertical samples** to simulate the 16-channel design of the VLP-16
- **Range from -15° to +15°** vertically to match the VLP-16 specifications
- **Range limits** from 0.2m to 100m with 1cm resolution
- **Gaussian noise model** with 1cm standard deviation
- **ROS integration** to publish sensor data as LaserScan messages

## Sensor Fusion Considerations

In realistic robotic applications, multiple sensors are often combined to provide more robust perception. The simulation should account for:

- **Time synchronization** between different sensor modalities
- **Coordinate frame transformations** between sensor frames
- **Noise correlation** between sensors mounted on the same platform
- **Computational requirements** of simulating multiple high-frequency sensors

## Summary

This sensor simulation guide has provided comprehensive coverage of LiDAR, IMU, and camera sensor modeling in robotics simulation environments. The ray casting theory and Velodyne VLP-16 simulation demonstrate the mathematical principles underlying LiDAR sensors. The IMU noise modeling explains how accelerometer and gyroscope measurements include various noise sources that must be accurately simulated. Camera simulation covers RGB-D and optical flow capabilities essential for visual perception tasks. The complete SDF snippet for LiDAR implementation provides practical guidance for configuring sensors in Gazebo simulations. Understanding these sensor simulation techniques is crucial for developing realistic and accurate robotic perception systems that can bridge the reality gap between simulation and real-world deployment.