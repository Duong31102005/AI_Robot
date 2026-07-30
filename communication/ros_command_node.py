try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    Node = object

from utils.logger import get_logger

logger = get_logger("ROSCommandNode")

if ROS_AVAILABLE:
    class CommandNode(Node):
        """Node ROS 2 lắng nghe các câu lệnh từ chủ thể (/robot/command)."""
        def __init__(self):
            super().__init__('robot_command_node')

            self.subscription = self.create_subscription(
                String,
                '/robot/command',
                self.command_callback,
                10
            )

            self.get_logger().info('Robot command node ONLINE')

        def command_callback(self, msg):
            text = msg.data.strip()

            if not text:
                return

            self.get_logger().info(f'COMMAND RECEIVED: {text}')
            # Xử lý phân tích lệnh tại đây (e.g. đi tới phòng khách, quay trái, dừng lại, theo tôi,...)

    def run_ros_node(args=None):
        rclpy.init(args=args)
        node = CommandNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        node.destroy_node()
        rclpy.shutdown()
else:
    def run_ros_node(args=None):
        logger.error("Thư viện rclpy (ROS 2) chưa được cài đặt trong môi trường Python hiện tại.")
