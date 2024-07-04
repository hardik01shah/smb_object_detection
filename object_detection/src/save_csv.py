#!/usr/bin/env python
"""
Get object detection results and save the classes, and poses in the world_graph_msf frame
"""

import rospy
import numpy as np
import pandas as pd
from geometry_msgs.msg import PoseStamped, PointStamped, Point
from std_msgs.msg import Header
from object_detection_msgs.msg import ObjectDetectionInfo, ObjectDetectionInfoArray
from tf import TransformListener
from tf.transformations import euler_from_quaternion

class SaveCSV:
    def __init__(self):
        self.sub = rospy.Subscriber('/object_detector/detection_info', ObjectDetectionInfoArray, self.callback)
        self.tf = TransformListener()
        self.data = []
        self.header = ['class', 'conf', 'x', 'y', 'z']

        # DEBUG
        # Publish the object poses in the world_graph_msf frame
        self.pub = rospy.Publisher('/custom/poses', PointStamped, queue_size=10)

    def callback(self, msg):
        # Get the object detection results and save the classes, and poses in the world_graph_msf frame
        # DEBUG: Publish the object poses in the world_graph_msf frame

        for detection in msg.info:
            class_name = detection.class_id
            conf = detection.confidence
            x = detection.position.x
            y = detection.position.y
            z = detection.position.z

            # Transform the object pose from the camera_link frame to the world_graph_msf frame
            pose = PoseStamped()
            pose.header = Header()
            pose.header = msg.header
            pose.pose.position = detection.position
            pose = self.tf.transformPose('world_graph_msf', pose)
            x = pose.pose.position.x
            y = pose.pose.position.y
            z = pose.pose.position.z

            pub_msg = PointStamped()
            pub_msg.header = Header()
            pub_msg.header = pose.header
            pub_msg.point = pose.pose.position
            print(pose.header)
            # DEBUG
            # Publish the object poses in the world_graph_msf frame
            self.pub.publish(pub_msg)

            self.data.append([class_name, conf, x, y, z])

    def save(self):
        df = pd.DataFrame(self.data, columns=self.header)
        df.to_csv('object_detection.csv', index=False)
        
if __name__ == '__main__':
    rospy.init_node('save_csv')
    sc = SaveCSV()
    rospy.spin()
    sc.save()