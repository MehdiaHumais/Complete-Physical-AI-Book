# Writing Your First ROS 2 Package: A Complete Tutorial

## Prerequisites

Before starting this tutorial, you should have:
- ROS 2 Humble Hawksbill installed on your system
- Python 3.10+ environment properly configured
- Basic understanding of Python programming concepts
- Familiarity with terminal/command line operations
- Knowledge of Linux/Unix file system navigation
- Understanding of object-oriented programming concepts

## Creating Your First ROS 2 Package

Creating a ROS 2 package is the foundational step for developing any robotic application. A ROS 2 package organizes code, dependencies, launch files, and configuration into a manageable unit that can be easily shared, built, and executed.

### Step 1: Setting Up Your Workspace

Before creating a package, you need to establish a proper workspace structure. The workspace serves as the development environment where all your ROS 2 packages will reside.

```bash
# Create the workspace directory structure
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Source the ROS 2 installation to make ROS 2 commands available
source /opt/ros/humble/setup.bash
```

The workspace follows the ROS 2 convention where the `src` directory contains all source code packages. The `src` directory is where you'll create and manage your packages.

### Step 2: Using the Package Creation Tool

ROS 2 provides a convenient command-line tool to create packages with the correct structure and files:

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python my_first_robot_package
```

This command creates a new Python-based package named `my_first_robot_package`. The `--build-type ament_python` flag specifies that this is a Python package, which means it will use the `ament_python` build system for proper Python package installation.

The command generates the following directory structure:

```
my_first_robot_package/
├── my_first_robot_package/
│   ├── __init__.py          # Python package initializer
│   └── my_first_robot_package.py  # Main Python module
├── test/
│   ├── test_copyright.py
│   ├── test_flake8.py
│   └── test_pep257.py
├── package.xml              # Package manifest describing dependencies and metadata
├── setup.cfg                # Installation configuration
├── setup.py                 # Python setup file for the package
└── README.md                # Documentation file
```

### Step 3: Understanding Package Configuration Files

#### package.xml Analysis

The `package.xml` file is an XML manifest that describes your package to the ROS 2 build system and dependency manager:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_first_robot_package</name>
  <version>0.0.0</version>
  <description>Package description</description>
  <maintainer email="user@todo.todo">user</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

This file contains:
- **Metadata**: Package name, version, description, and maintainer information
- **Dependencies**: Runtime dependencies like `rclpy` (ROS 2 Python client library) and `std_msgs` (standard message types)
- **Test dependencies**: Tools for code quality checking
- **Build type**: Specifies the build system to use (ament_python)

#### setup.py Configuration

The `setup.py` file configures how Python packages are built and installed:

```python
from setuptools import setup

package_name = 'my_first_robot_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
```

This configuration:
- Defines the package name and version
- Specifies which Python packages to install
- Sets up data files needed by the ROS 2 system
- Configures entry points for executable scripts

## Complete Talker (Publisher) Implementation

The talker node demonstrates the publisher pattern in ROS 2, where it periodically publishes messages to a topic that other nodes can subscribe to.

### talker.py - Full Implementation

```python
#!/usr/bin/env python3
"""
Talker Node - Publisher Example

This node demonstrates the publisher pattern in ROS 2.
It periodically publishes messages to a topic called 'chatter'.
"""

# Import the core ROS 2 Python library
import rclpy
from rclpy.node import Node

# Import standard message types for communication
from std_msgs.msg import String


class TalkerNode(Node):
    """
    A ROS 2 node that publishes messages to a topic.
    
    This class inherits from rclpy.node.Node and implements
    the publisher functionality using ROS 2's publish-subscribe model.
    """
    
    def __init__(self):
        """
        Initialize the TalkerNode instance.
        
        This method sets up the publisher, timer, and internal counter.
        The super().__init__() call initializes the parent Node class.
        """
        # Call the parent class (Node) constructor with the node name
        # This registers the node with the ROS 2 graph and initializes ROS communications
        super().__init__('talker_node')
        
        # Create a publisher that will publish String messages
        # Parameters:
        # - Message type: std_msgs.msg.String - defines the message structure
        # - Topic name: 'chatter' - the name of the topic to publish to
        # - Queue size: 10 - the number of messages to buffer if the subscriber is slow
        self.publisher = self.create_publisher(
            String,         # Message type to publish
            'chatter',      # Topic name
            10              # Queue size for outgoing messages
        )
        
        # Create a timer that calls the publish_message method every 0.5 seconds
        # This enables periodic publishing without blocking the main thread
        self.timer = self.create_timer(0.5, self.publish_message)
        
        # Initialize a counter to track published messages
        self.counter = 0
        
        # Log a message indicating successful node initialization
        self.get_logger().info('Talker node initialized successfully')

    def publish_message(self):
        """
        Publish a message to the 'chatter' topic.
        
        This method is called by the timer every 0.5 seconds.
        It creates a String message with the current counter value and publishes it.
        """
        # Create a new String message instance
        # This is the message object that will be sent to subscribers
        msg = String()
        
        # Set the message data to include the current counter value
        # This creates a unique message for each publication
        msg.data = f'Hello ROS 2 World: {self.counter}'
        
        # Publish the message to the 'chatter' topic
        # This sends the message to all subscribers of this topic
        self.publisher.publish(msg)
        
        # Log the published message to the console for debugging
        self.get_logger().info(f'Published message: "{msg.data}"')
        
        # Increment the counter for the next message
        self.counter += 1


def main(args=None):
    """
    Main function to run the Talker node.
    
    This function follows the standard ROS 2 Python node pattern:
    1. Initialize ROS 2 communications
    2. Create the node instance
    3. Run the event loop (spin)
    4. Clean up resources when done
    
    Args:
        args: Command line arguments (typically None)
    """
    # Initialize the ROS 2 client library
    # This must be called before creating any nodes
    # It initializes the underlying middleware and communication system
    rclpy.init(args=args)
    
    # Create an instance of the TalkerNode class
    # This registers the node with the ROS 2 graph and sets up communication channels
    talker_node = TalkerNode()
    
    try:
        # Start the event loop (spin)
        # This keeps the node running and processes incoming/outgoing messages
        # The loop runs indefinitely until interrupted (Ctrl+C) or shutdown
        rclpy.spin(talker_node)
    except KeyboardInterrupt:
        # Handle keyboard interrupt (Ctrl+C) gracefully
        # This allows for clean shutdown when user interrupts the process
        pass
    finally:
        # Clean up resources regardless of how the loop exits
        # Destroy the node to release resources and unregister from ROS graph
        talker_node.destroy_node()
        
        # Shutdown the ROS 2 client library
        # This cleans up all ROS communications and releases system resources
        rclpy.shutdown()


# Standard Python idiom to ensure the main function is only called
# when this script is executed directly (not imported as a module)
if __name__ == '__main__':
    # Call the main function when the script is run directly
    main()
```

## Complete Listener (Subscriber) Implementation

The listener node demonstrates the subscriber pattern in ROS 2, where it receives messages from a topic published by other nodes.

### listener.py - Full Implementation

```python
#!/usr/bin/env python3
"""
Listener Node - Subscriber Example

This node demonstrates the subscriber pattern in ROS 2.
It subscribes to messages from the 'chatter' topic and logs them.
"""

# Import the core ROS 2 Python library
import rclpy
from rclpy.node import Node

# Import standard message types for communication
from std_msgs.msg import String


class ListenerNode(Node):
    """
    A ROS 2 node that subscribes to messages from a topic.
    
    This class inherits from rclpy.node.Node and implements
    the subscriber functionality using ROS 2's publish-subscribe model.
    """
    
    def __init__(self):
        """
        Initialize the ListenerNode instance.
        
        This method sets up the subscription to receive messages.
        The super().__init__() call initializes the parent Node class.
        """
        # Call the parent class (Node) constructor with the node name
        # This registers the node with the ROS 2 graph and initializes ROS communications
        super().__init__('listener_node')
        
        # Create a subscription to receive messages from the 'chatter' topic
        # Parameters:
        # - Message type: std_msgs.msg.String - defines the expected message structure
        # - Topic name: 'chatter' - the topic to subscribe to
        # - Callback function: self.listener_callback - called when messages arrive
        # - Queue size: 10 - the number of messages to buffer if the callback is slow
        self.subscription = self.create_subscription(
            String,                 # Expected message type
            'chatter',              # Topic name to subscribe to
            self.listener_callback, # Callback function for incoming messages
            10                      # Queue size for incoming messages
        )
        
        # Keep a reference to the subscription to prevent garbage collection
        # Without this reference, Python's garbage collector might remove the subscription
        # This is necessary to keep the subscription active during node execution
        self.subscription  # This line prevents an unused variable warning
        
        # Log a message indicating successful node initialization
        self.get_logger().info('Listener node initialized successfully')

    def listener_callback(self, msg):
        """
    Callback function for processing incoming messages.
        
        This method is called whenever a new message arrives on the subscribed topic.
        It receives the message as a parameter and processes it accordingly.
        
        Args:
            msg: The incoming message of type std_msgs.msg.String
        """
        # Log the received message data to the console for monitoring
        # This demonstrates that the message was successfully received
        self.get_logger().info(f'Received message: "{msg.data}"')


def main(args=None):
    """
    Main function to run the Listener node.
    
    This function follows the standard ROS 2 Python node pattern:
    1. Initialize ROS 2 communications
    2. Create the node instance
    3. Run the event loop (spin)
    4. Clean up resources when done
    
    Args:
        args: Command line arguments (typically None)
    """
    # Initialize the ROS 2 client library
    # This must be called before creating any nodes
    # It initializes the underlying middleware and communication system
    rclpy.init(args=args)
    
    # Create an instance of the ListenerNode class
    # This registers the node with the ROS 2 graph and sets up communication channels
    listener_node = ListenerNode()
    
    try:
        # Start the event loop (spin)
        # This keeps the node running and processes incoming messages
        # The loop runs indefinitely until interrupted (Ctrl+C) or shutdown
        rclpy.spin(listener_node)
    except KeyboardInterrupt:
        # Handle keyboard interrupt (Ctrl+C) gracefully
        # This allows for clean shutdown when user interrupts the process
        pass
    finally:
        # Clean up resources regardless of how the loop exits
        # Destroy the node to release resources and unregister from ROS graph
        listener_node.destroy_node()
        
        # Shutdown the ROS 2 client library
        # This cleans up all ROS communications and releases system resources
        rclpy.shutdown()


# Standard Python idiom to ensure the main function is only called
# when this script is executed directly (not imported as a module)
if __name__ == '__main__':
    # Call the main function when the script is run directly
    main()
```

## Complete Build and Execution Process

### Step 1: Update setup.py for Executable Scripts

Before building, you need to register your talker and listener nodes as executable scripts in the `setup.py` file:

```python
from setuptools import setup

package_name = 'my_first_robot_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = my_first_robot_package.talker:main',
            'listener = my_first_robot_package.listener:main',
        ],
    },
)
```

The `entry_points` section registers the talker and listener functions as console scripts, making them executable through ROS 2's command-line tools.

### Step 2: Building the Package with colcon

The `colcon` build tool is ROS 2's preferred build system. It handles the compilation and installation of packages:

```bash
# Navigate to the workspace root directory
cd ~/ros2_ws

# Source the ROS 2 installation to ensure build tools are available
source /opt/ros/humble/setup.bash

# Build only the specific package to save time
colcon build --packages-select my_first_robot_package

# Alternative: Build all packages in the workspace
# colcon build
```

The `colcon build` command performs several important steps:

1. **Dependency Resolution**: Analyzes package.xml to determine dependencies
2. **Code Generation**: Generates necessary code for message and service types
3. **Compilation**: Compiles Python packages and any C++ code
4. **Installation**: Installs the package to the `install` directory
5. **Setup Files**: Generates setup files for sourcing the built package

### Step 3: Sourcing the Built Package

After building, you need to source the installation setup to make the built package available in your current shell:

```bash
# Source the installation setup
source install/setup.bash

# Alternative: Add to your .bashrc to make it permanent
# echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

This step adds the built package to your ROS 2 environment, making the executable scripts available through ROS 2 commands.

### Step 4: Running the Nodes

Now you can run your nodes using ROS 2's command-line tools:

```bash
# Terminal 1: Run the talker node
source ~/ros2_ws/install/setup.bash
ros2 run my_first_robot_package talker

# Terminal 2: Run the listener node
source ~/ros2_ws/install/setup.bash
ros2 run my_first_robot_package listener
```

## Line-by-Line Code Analysis

### Talker Node Analysis

**Lines 1-3**: `#!/usr/bin/env python3` is the shebang that specifies the Python interpreter to use.

**Lines 6-8**: Import statements bring in required ROS 2 modules:
- `rclpy`: The ROS 2 Python client library
- `Node`: The base class for ROS 2 nodes
- `String`: A standard message type for text data

**Lines 11-40**: The `TalkerNode` class definition:
- Inherits from `Node` class, gaining ROS 2 functionality
- Constructor (`__init__`) method initializes the node, publisher, timer, and counter
- `self.create_publisher()` creates a publisher with message type, topic name, and queue size
- `self.create_timer()` sets up periodic execution of the publish method
- `publish_message()` method creates and publishes String messages

**Lines 43-61**: The `main` function:
- `rclpy.init()` initializes the ROS 2 client library
- Creates an instance of the `TalkerNode`
- `rclpy.spin()` starts the event loop that processes callbacks
- `destroy_node()` and `rclpy.shutdown()` handle cleanup

**Lines 64-67**: The `if __name__ == '__main__':` guard ensures the main function runs only when the script is executed directly.

### Listener Node Analysis

**Lines 11-39**: The `ListenerNode` class:
- Inherits from `Node` class
- Constructor creates a subscription using `self.create_subscription()`
- `listener_callback()` method processes incoming messages

**Lines 41-61**: The `main` function mirrors the talker's pattern with appropriate cleanup for the listener node.

## Understanding Key Concepts

### Classes and Inheritance

The talker and listener nodes inherit from the `Node` class, which provides:
- Node lifecycle management
- Communication interfaces (publishers, subscribers, services, actions)
- Logging functionality
- Parameter management
- Clock and timer functionality

### The Spin Loop

The `rclpy.spin()` function implements an event loop that:
- Processes incoming messages and calls appropriate callbacks
- Handles service requests
- Processes action requests
- Maintains node lifecycle
- Continues running until interrupted or explicitly stopped

### Publisher-Subscriber Pattern

The publisher-subscriber pattern enables asynchronous communication:
- Publishers send messages without knowing about subscribers
- Subscribers receive messages without knowing about publishers
- Topics act as communication channels
- Multiple publishers and subscribers can use the same topic

## Advanced Execution Considerations

### Running Multiple Instances

You can run multiple instances of the same node with different namespaces:

```bash
# Terminal 1
ros2 run my_first_robot_package talker --ros-args -r __ns:=/robot1

# Terminal 2
ros2 run my_first_robot_package listener --ros-args -r __ns:=/robot1
```

### Parameter Passing

Nodes can accept parameters from the command line:

```bash
# Pass parameters to the node
ros2 run my_first_robot_package talker --ros-args -p publish_rate:=2.0
```

### Launch Files

For more complex applications, ROS 2 launch files can start multiple nodes simultaneously:

```xml
<launch>
  <node pkg="my_first_robot_package" exec="talker" name="talker_node"/>
  <node pkg="my_first_robot_package" exec="listener" name="listener_node"/>
</launch>
```

## Common Troubleshooting

### Build Issues

If the build fails, check:
- Correct Python version (3.10+)
- ROS 2 Humble properly sourced
- Package name matches directory structure
- Dependencies properly declared in package.xml

### Runtime Issues

If nodes don't communicate:
- Ensure both nodes are run from sourced terminals
- Verify topic names match exactly
- Check QoS policies are compatible
- Confirm nodes are on the same ROS domain

## Summary

This comprehensive tutorial has walked through the complete process of creating your first ROS 2 package, from initial setup through implementation and execution. You've learned to create publisher and subscriber nodes using the `rclpy` library, with detailed explanations of each line of code.

The talker and listener example demonstrates the fundamental publish-subscribe communication pattern that underlies most ROS 2 applications. Understanding these concepts provides the foundation for building more complex robotic systems that leverage ROS 2's distributed architecture.

The build process using `colcon` and the execution workflow with `ros2 run` are essential skills for any ROS 2 developer. Mastering these fundamentals enables you to create, build, and run ROS 2 packages for increasingly sophisticated robotic applications.