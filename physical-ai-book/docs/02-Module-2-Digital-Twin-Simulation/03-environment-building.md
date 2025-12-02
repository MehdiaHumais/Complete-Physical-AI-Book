# World Building Tutorial: Creating Realistic Simulation Environments

## Prerequisites

Before diving into this module, students should have:
- Basic understanding of 3D modeling concepts and coordinate systems
- Familiarity with XML syntax and structure
- Knowledge of mesh file formats (DAE, OBJ)
- Experience with Gazebo simulation environment
- Understanding of SDF (Simulation Description Format) basics
- Blender or similar 3D modeling software knowledge (optional)

## Asset Integration: Importing Blender Meshes into Gazebo

Creating realistic simulation environments often requires importing custom 3D models created in external tools like Blender. Gazebo supports several mesh formats, with COLLADA (.dae) being the preferred format due to its comprehensive support for materials, textures, and geometry.

### Blender to Gazebo Workflow

The process of importing Blender meshes into Gazebo involves several steps to ensure proper scaling, materials, and physics properties:

1. **Model Creation in Blender**: Create your 3D model with appropriate scale (1 unit = 1 meter in Gazebo)
2. **Material Assignment**: Assign materials with proper diffuse, specular, and normal maps
3. **Export Settings**: Export as COLLADA format with specific configurations
4. **File Organization**: Structure files properly in Gazebo's model directory

### Exporting from Blender

When exporting from Blender to Gazebo, follow these critical settings:

```python
# Blender export configuration for Gazebo compatibility
blender_export_settings = {
    'filepath': 'model.dae',
    'use_selection': False,
    'use_mesh_modifiers': True,
    'use_tamp': False,  # Disable tamp to avoid scaling issues
    'use_rot': False,   # Disable rotation to maintain coordinate system
    'use_scal': True,   # Preserve scale information
    'copy_images': True,
    'use_texture_copies': True
}
```

### Coordinate System Considerations

Blender uses a Z-up coordinate system, while Gazebo uses a Z-up system as well, but with different conventions for rotations. Ensure your models are properly oriented before export:

- **Position**: Models should be at world origin or positioned correctly
- **Scale**: 1 Blender unit = 1 meter in Gazebo
- **Rotation**: Apply rotations (Ctrl+A in Blender) before export to avoid transformation issues

### Model File Structure

Gazebo expects a specific directory structure for custom models:

```
~/.gazebo/models/my_model/
├── model.sdf
├── mesh/
│   ├── visual.dae
│   └── collision.dae
├── materials/
│   ├── textures/
│   │   ├── texture1.png
│   │   └── texture2.jpg
│   └── scripts/
│       └── model.material
└── model.config
```

### Sample model.config for Asset Loading

```xml
<?xml version="1.0"?>
<model>
  <name>custom_furniture</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
  <author>
    <name>Your Name</name>
    <email>your.email@example.com</email>
  </author>
  <description>A custom furniture model for home simulation.</description>
</model>
```

## SDF Format: Deep Dive into Simulation Description Format Tags

The Simulation Description Format (SDF) is an XML-based format that describes environments, robots, and objects in Gazebo. Understanding SDF tags is crucial for creating complex simulation worlds.

### Core SDF Structure

An SDF file follows this hierarchical structure:

```xml
<sdf version="1.6">
  <world name="world_name">
    <!-- World elements -->
  </world>
  
  <model name="model_name">
    <!-- Model elements -->
  </model>
  
  <light name="light_name">
    <!-- Light elements -->
  </light>
  
  <actor name="actor_name">
    <!-- Actor elements -->
  </actor>
</sdf>
```

### World-Level Tags

**`<gravity>`**: Defines gravitational acceleration in m/s²:

```xml
<gravity>0 0 -9.8</gravity>
```

**`<physics>`**: Configures physics engine parameters:

```xml
<physics name="default_physics" type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>
    </solver>
  </ode>
</physics>
```

### Model-Level Tags

**`<link>`**: Represents a rigid body with collision, visual, and inertial properties:

```xml
<link name="link_name">
  <!-- Visual properties -->
  <visual name="visual">
    <geometry>
      <mesh>
        <uri>model://my_model/meshes/part.dae</uri>
        <scale>1 1 1</scale>
      </mesh>
    </geometry>
    <material>
      <script>
        <uri>file://media/materials/scripts/gazebo.material</uri>
        <name>Gazebo/Blue</name>
      </script>
    </material>
  </visual>
  
  <!-- Collision properties -->
  <collision name="collision">
    <geometry>
      <mesh>
        <uri>model://my_model/meshes/part.dae</uri>
        <scale>1 1 1</scale>
      </mesh>
    </geometry>
    <surface>
      <friction>
        <ode>
          <mu>1.0</mu>
          <mu2>1.0</mu2>
        </ode>
      </friction>
    </surface>
  </collision>
  
  <!-- Inertial properties -->
  <inertial>
    <mass>1.0</mass>
    <inertia>
      <ixx>0.1</ixx>
      <ixy>0</ixy>
      <ixz>0</ixz>
      <iyy>0.1</iyy>
      <iyz>0</iyz>
      <izz>0.1</izz>
    </inertia>
  </inertial>
</link>
```

### Joint-Level Tags

**`<joint>`**: Defines connections between links:

```xml
<joint name="joint_name" type="revolute">
  <parent>parent_link</parent>
  <child>child_link</child>
  <axis>
    <xyz>0 0 1</xyz>
    <limit>
      <lower>-1.57</lower>
      <upper>1.57</upper>
      <effort>100</effort>
      <velocity>3.0</velocity>
    </limit>
  </axis>
</joint>
```

### Sensor Configuration Tags

**`<sensor>`**: Defines various sensor types:

```xml
<sensor name="camera" type="camera">
  <always_on>1</always_on>
  <update_rate>30</update_rate>
  <camera>
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>100</far>
    </clip>
  </camera>
</sensor>
```

## Step-by-Step Home Environment Tutorial

Creating a realistic home environment requires careful planning and implementation of architectural elements, furniture, and interactive objects.

### Step 1: Create the Basic Room Structure

Start by creating the fundamental room layout using geometric primitives:

```xml
<?xml version="1.0"?>
<sdf version="1.7">
  <world name="home_environment">
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
      <direction>-0.3 0.1 -0.9</direction>
    </light>
    
    <!-- Floor -->
    <model name="floor">
      <static>true</static>
      <link name="floor_link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 10 0.1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 10 0.1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.3 0.3 0.3 1</ambient>
            <diffuse>0.3 0.3 0.3 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- Walls -->
    <model name="wall_front">
      <static>true</static>
      <pose>0 5 1.5 0 0 0</pose>
      <link name="wall_link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 3</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 3</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- Continue with side walls and back wall -->
    <model name="wall_left">
      <static>true</static>
      <pose>-5 0 1.5 0 0 1.57</pose>
      <link name="wall_link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 3</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 3</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <model name="wall_right">
      <static>true</static>
      <pose>5 0 1.5 0 0 -1.57</pose>
      <link name="wall_link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 3</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 3</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <model name="wall_back">
      <static>true</static>
      <pose>0 -5 1.5 0 0 3.14</pose>
      <link name="wall_link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 3</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 3</size>
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

### Step 2: Add Furniture Elements

Create furniture models using either primitive shapes or imported meshes:

```xml
<!-- Living room furniture -->
<model name="sofa">
  <pose>-2 -2 0.3 0 0 0</pose>
  <link name="sofa_base">
    <collision name="collision">
      <geometry>
        <box>
          <size>2.0 0.8 0.6</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>2.0 0.8 0.6</size>
        </box>
      </geometry>
      <material>
        <ambient>0.4 0.2 0.1 1</ambient>
        <diffuse>0.7 0.4 0.2 1</diffuse>
        <specular>0.1 0.1 0.1 1</specular>
      </material>
    </visual>
  </link>
  
  <!-- Back rest -->
  <link name="sofa_back">
    <pose>0 0 0.3 0 0 0</pose>
    <collision name="collision">
      <geometry>
        <box>
          <size>2.0 0.1 0.6</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>2.0 0.1 0.6</size>
        </box>
      </geometry>
      <material>
        <ambient>0.4 0.2 0.1 1</ambient>
        <diffuse>0.7 0.4 0.2 1</diffuse>
      </material>
    </visual>
  </link>
  
  <!-- Connection between base and back rest -->
  <joint name="sofa_back_joint" type="fixed">
    <parent>sofa_base</parent>
    <child>sofa_back</child>
    <pose>0 0 0.3 0 0 0</pose>
  </joint>
</model>

<!-- Coffee table -->
<model name="coffee_table">
  <pose>0 -1.5 0.3 0 0 0</pose>
  <link name="table_top">
    <collision name="collision">
      <geometry>
        <box>
          <size>1.0 0.6 0.05</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>1.0 0.6 0.05</size>
        </box>
      </geometry>
      <material>
        <ambient>0.6 0.4 0.2 1</ambient>
        <diffuse>0.8 0.6 0.4 1</diffuse>
      </material>
    </visual>
  </link>
  
  <!-- Table legs -->
  <link name="leg1">
    <pose>-0.4 -0.25 0.25 0 0 0</pose>
    <collision name="collision">
      <geometry>
        <cylinder>
          <radius>0.03</radius>
          <length>0.5</length>
        </cylinder>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <cylinder>
          <radius>0.03</radius>
          <length>0.5</length>
        </cylinder>
      </geometry>
      <material>
        <ambient>0.6 0.4 0.2 1</ambient>
        <diffuse>0.8 0.6 0.4 1</diffuse>
      </material>
    </visual>
  </link>
  
  <!-- Additional legs and connections would go here -->
</model>
```

### Step 3: Add Interactive Elements

Include elements that robots can interact with:

```xml
<!-- Interactive door -->
<model name="door">
  <pose>3 4.9 1.2 0 0 0</pose>
  <link name="door_frame">
    <collision name="collision">
      <geometry>
        <box>
          <size>0.1 2.4 1.8</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>0.1 2.4 1.8</size>
        </box>
      </geometry>
      <material>
        <ambient>0.5 0.5 0.5 1</ambient>
        <diffuse>0.7 0.7 0.7 1</diffuse>
      </material>
    </visual>
  </link>
  
  <link name="door_panel">
    <collision name="collision">
      <geometry>
        <box>
          <size>0.05 2.4 1.8</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>0.05 2.4 1.8</size>
        </box>
      </geometry>
      <material>
        <ambient>0.8 0.6 0.4 1</ambient>
        <diffuse>0.9 0.7 0.5 1</diffuse>
      </material>
    </visual>
  </link>
  
  <joint name="door_hinge" type="revolute">
    <parent>door_frame</parent>
    <child>door_panel</child>
    <axis>
      <xyz>0 1 0</xyz>
      <limit>
        <lower>-1.57</lower>
        <upper>0</upper>
      </limit>
    </axis>
    <pose>-0.025 0 0 0 0 0</pose>
  </joint>
</model>
```

## Lighting Setup for Realism

Proper lighting is crucial for creating realistic simulation environments and ensuring computer vision algorithms work correctly.

### Directional Light Configuration

The sun is typically implemented as a directional light that casts parallel rays:

```xml
<light name="sun" type="directional">
  <pose>0 0 10 0 0 0</pose>
  <diffuse>0.8 0.8 0.8 1</diffuse>
  <specular>0.2 0.2 0.2 1</specular>
  
  <!-- Direction vector pointing from light to scene -->
  <direction>-0.3 0.1 -0.9</direction>
  
  <!-- Enable shadows for realism -->
  <cast_shadows>true</cast_shadows>
  
  <!-- Attenuation affects how light intensity drops with distance -->
  <attenuation>
    <range>1000</range>
    <constant>0.9</constant>
    <linear>0.01</linear>
    <quadratic>0.001</quadratic>
  </attenuation>
</light>
```

### Shadow Mapping

To enhance realism, configure shadow parameters:

```xml
<scene>
  <shadows>true</shadows>
  <ambient>0.3 0.3 0.3 1</ambient>
  <background>0.7 0.7 0.7 1</background>
</scene>

<!-- In the light definition -->
<light name="main_light">
  <cast_shadows>true</cast_shadows>
  <shadow>
    <clip>
      <near>0.1</near>
      <far>50.0</far>
    </clip>
    <cull_face>back</cull_face>
  </shadow>
</light>
```

### Multiple Light Sources

For more complex lighting scenarios, combine multiple light types:

```xml
<!-- Ambient light to fill shadows -->
<light name="ambient_fill" type="point">
  <pose>0 0 5 0 0 0</pose>
  <diffuse>0.3 0.3 0.3 1</diffuse>
  <attenuation>
    <range>20</range>
    <constant>0.2</constant>
    <linear>0.05</linear>
    <quadratic>0.01</quadratic>
  </attenuation>
</light>

<!-- Table lamp -->
<light name="table_lamp" type="spot">
  <pose>-1.5 -1 1.5 0 0 0</pose>
  <diffuse>0.9 0.9 0.7 1</diffuse>
  <specular>0.9 0.9 0.7 1</specular>
  <direction>0 0 -1</direction>
  <spot>
    <inner_angle>0.1</inner_angle>
    <outer_angle>0.5</outer_angle>
    <falloff>1.0</falloff>
  </spot>
  <attenuation>
    <range>5</range>
  </attenuation>
</light>
```

### Dynamic Lighting Considerations

For realistic home environments, consider implementing time-of-day variations:

```python
# Python script to dynamically change lighting
import math

def calculate_sun_direction(time_of_day_hours):
    """
    Calculate sun direction based on time of day
    Time: 0 = midnight, 6 = sunrise, 12 = noon, 18 = sunset
    """
    # Convert to radians (12 hours = π radians)
    angle = (time_of_day_hours - 6) * math.pi / 12
    
    # Calculate sun direction vector
    elevation = math.sin(angle) * 0.7  # Maximum elevation of 0.7
    azimuth = math.cos(angle) * 0.7
    
    # Normalize and return direction vector
    length = math.sqrt(elevation**2 + azimuth**2 + 0.5**2)
    direction = [
        azimuth / length,
        0.1 / length,  # Small northward component
        elevation / length
    ]
    
    return direction
```

## Performance Optimization

For large environments, consider these performance optimizations:

- **Level of Detail (LOD)**: Use simpler models for distant objects
- **Occlusion Culling**: Hide objects not visible from camera view
- **Texture Compression**: Use compressed textures to reduce memory usage
- **Collision Simplification**: Use simplified collision meshes for complex visual models

## Summary

This comprehensive world building tutorial has covered the essential aspects of creating realistic simulation environments in Gazebo. We've explored the process of importing Blender meshes with proper configuration and file structure. The deep dive into SDF format tags provides the foundation for creating complex simulation elements. The step-by-step home environment tutorial demonstrates practical implementation of architectural elements, furniture, and interactive components. Finally, the lighting setup section explains how to create realistic illumination with proper shadows and multiple light sources. These skills are essential for creating simulation environments that accurately represent real-world scenarios for robotic testing and development.