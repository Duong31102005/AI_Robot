import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from communication.ros_command_node import run_ros_node

if __name__ == "__main__":
    run_ros_node()
