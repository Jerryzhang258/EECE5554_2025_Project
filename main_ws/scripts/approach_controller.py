#!/usr/bin/env python3
"""
Approach Controller Node - Uses RSyncDetectionList and RoboSync
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

from object_location_interfaces.msg import RSyncDetectionList, DetectedItem, RoboSync

class ApproachControllerNode(Node):
    def __init__(self):
        super().__init__('approach_controller_node')
        
        # Parameters
        self.declare_parameter('target_class', 'bottle')
        self.declare_parameter('target_distance', 0.4)
        self.declare_parameter('max_linear_speed', 0.15)
        self.declare_parameter('max_angular_speed', 0.5)
        self.declare_parameter('kp_angular', 0.004)
        self.declare_parameter('kp_linear', 0.5)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('distance_tolerance', 0.05)
        self.declare_parameter('angular_tolerance', 30)
        self.declare_parameter('enabled', True)
        self.declare_parameter('timeout', 10.0)
        
        self.target_class = self.get_parameter('target_class').value
        self.target_distance = self.get_parameter('target_distance').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value
        self.kp_angular = self.get_parameter('kp_angular').value
        self.kp_linear = self.get_parameter('kp_linear').value
        self.image_width = self.get_parameter('image_width').value
        self.distance_tolerance = self.get_parameter('distance_tolerance').value
        self.angular_tolerance = self.get_parameter('angular_tolerance').value
        self.enabled = self.get_parameter('enabled').value
        self.timeout = self.get_parameter('timeout').value
        
        self.image_center_x = self.image_width / 2
        self.bridge = CvBridge()
        
        # State
        self.current_detection = None
        self.current_depth_image = None
        self.last_detection_time = self.get_clock().now()
        self.approaching = False
        self.target_reached = False
        
        # Subscribers
        self.detection_sub = self.create_subscription(
            RSyncDetectionList,
            '/objects/detections',
            self.detection_callback,
            10
        )
        
        self.sync_sub = self.create_subscription(
            RoboSync,
            '/sync/robot/state',
            self.sync_callback,
            10
        )
        
        # ✅ 修复：直接发布到底盘订阅的话题
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel_unstamped', 10)
        
        # Control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('='*60)
        self.get_logger().info('Approach Controller Started')
        self.get_logger().info(f'Target: {self.target_class} | Distance: {self.target_distance}m')
        self.get_logger().info('='*60)
    
    def detection_callback(self, msg):
        try:
            target_found = False
            
            for item in msg.detections.item_list:
                if item.name == self.target_class:
                    if len(item.xywh) >= 2:
                        center_x = float(item.xywh[0])
                        center_y = float(item.xywh[1])
                        
                        self.current_detection = {
                            'center_x': center_x,
                            'center_y': center_y,
                            'name': item.name,
                            'confidence': item.confidence
                        }
                        
                        self.last_detection_time = self.get_clock().now()
                        target_found = True
                        
                        if not self.approaching:
                            self.get_logger().info(
                                f'🎯 {item.name} detected at ({center_x:.0f}, {center_y:.0f})'
                            )
                        self.approaching = True
                        break
            
            if not target_found:
                if self.approaching:
                    self.get_logger().info(f'Lost {self.target_class}')
                self.current_detection = None
                self.approaching = False
                
        except Exception as e:
            self.get_logger().error(f'Detection error: {str(e)}')
    
    def sync_callback(self, msg):
        try:
            self.current_depth_image = msg.depth_image
        except Exception as e:
            self.get_logger().error(f'Sync error: {str(e)}')
    
    def get_distance_from_depth(self, center_x, center_y):
        if self.current_depth_image is None:
            return None
        
        try:
            depth_array = self.bridge.imgmsg_to_cv2(
                self.current_depth_image,
                desired_encoding='passthrough'
            )
            
            height, width = depth_array.shape
            cx = int(np.clip(center_x, 0, width - 1))
            cy = int(np.clip(center_y, 0, height - 1))
            
            sample_size = 5
            x_start = max(0, cx - sample_size)
            x_end = min(width, cx + sample_size)
            y_start = max(0, cy - sample_size)
            y_end = min(height, cy + sample_size)
            
            depth_region = depth_array[y_start:y_end, x_start:x_end]
            valid_depths = depth_region[(depth_region > 0) & (~np.isnan(depth_region))]
            
            if len(valid_depths) > 0:
                median_depth = np.median(valid_depths)
                distance = median_depth / 1000.0
                
                if 0.2 < distance < 5.0:
                    return distance
            
            return None
            
        except Exception as e:
            self.get_logger().error(f'Depth error: {str(e)}')
            return None
    
    def control_loop(self):
        if not self.enabled:
            return
        
        time_since = (self.get_clock().now() - self.last_detection_time).nanoseconds / 1e9
        
        if time_since > self.timeout:
            if self.approaching:
                self.get_logger().info('⏱️ Timeout, stopping', throttle_duration_sec=2.0)
                self.stop_robot()
                self.approaching = False
                self.target_reached = False
            return
        
        if self.current_detection is not None and self.approaching:
            distance = self.get_distance_from_depth(
                self.current_detection['center_x'],
                self.current_detection['center_y']
            )
            
            if distance is not None:
                self.approach_target(self.current_detection['center_x'], distance)
            else:
                self.get_logger().warn('No valid depth', throttle_duration_sec=2.0)
    
    def approach_target(self, target_x, distance):
        twist = Twist()
        
        error_x = target_x - self.image_center_x
        angular_vel = -self.kp_angular * error_x
        angular_vel = np.clip(angular_vel, -self.max_angular_speed, self.max_angular_speed)
        
        distance_error = distance - self.target_distance
        
        if abs(distance_error) <= self.distance_tolerance and abs(error_x) <= self.angular_tolerance:
            if not self.target_reached:
                self.get_logger().info('✅ TARGET REACHED!')
                self.target_reached = True
            
            twist.linear.x = 0.0
            twist.angular.z = angular_vel * 0.3
        else:
            self.target_reached = False
            
            if abs(distance_error) > self.distance_tolerance:
                linear_vel = self.kp_linear * distance_error
                
                if abs(error_x) > 100:
                    linear_vel *= 0.3
                elif abs(error_x) > 50:
                    linear_vel *= 0.5
                
                linear_vel = np.clip(linear_vel, -self.max_linear_speed, self.max_linear_speed)
            else:
                linear_vel = 0.0
            
            twist.linear.x = linear_vel
            twist.angular.z = angular_vel
            
            self.get_logger().info(
                f'→ D={distance:.2f}m | err={distance_error:+.2f}m | '
                f'angle_err={error_x:+.0f}px | v={linear_vel:.2f} w={angular_vel:.2f}',
                throttle_duration_sec=1.0
            )
        
        try:
            self.cmd_vel_pub.publish(twist)
        except:
            pass
    
    def stop_robot(self):
        """Stop the robot"""
        try:
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
        except:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = ApproachControllerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    except:
        pass
    finally:
        try:
            node.stop_robot()
        except:
            pass
        try:
            node.destroy_node()
        except:
            pass
        try:
            rclpy.shutdown()
        except:
            pass

if __name__ == '__main__':
    main()
