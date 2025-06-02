import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
import time

class MyNodes(Node):
    def __init__(self):
        super().__init__('follow_node')
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.subscription_center = self.create_subscription(
            Int32MultiArray,
            '/object_center',
            self.center_callback,
            10)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        self.image_width = None
        self.object_center_x = None
        self.last_object_time = None
        
        self.turn_speed = 0.2
        self.center_tolerance = 30  # pixels
        self.timeout_sec = 1.0      # seconds without update = lost object

        self.timer = self.create_timer(0.1, self.control_loop)
        
    def image_callback(self, msg):
        self.image_width = msg.width

    def center_callback(self, msg):
        if len(msg.data) >= 2:
            self.object_center_x = msg.data[0]
            self.last_object_time = time.time()
        else:
            #self.last_pos = self.object_center_x
            self.object_center_x = None

    def control_loop(self):
        cmd = Twist()
        current_time = time.time()

        object_visible = (
            self.object_center_x is not None and 
            self.last_object_time is not None and 
            (current_time - self.last_object_time) < self.timeout_sec
        )

        if not object_visible or self.image_width is None:
            cmd.angular.z = self.turn_speed 
            
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = MyNodes()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
