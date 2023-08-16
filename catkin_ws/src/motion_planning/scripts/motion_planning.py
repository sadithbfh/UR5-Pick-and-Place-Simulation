#!/usr/bin/python3

import os
import math
import copy
import json
import actionlib
import control_msgs.msg
from controller import ArmController
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float32MultiArray
import rospy
from geometry_msgs.msg import *
from pyquaternion import Quaternion as PyQuaternion
import numpy as np
from gazebo_ros_link_attacher.srv import SetStatic, SetStaticRequest, SetStaticResponse
from gazebo_ros_link_attacher.srv import Attach, AttachRequest, AttachResponse
import tf.transformations as tft
import tf.transformations as tf_trans

from helpers.transforms import current_robot_pose, publish_tf_quaterion_as_transform, convert_pose, publish_pose_as_transform


PKG_PATH = os.path.dirname(os.path.abspath(__file__))

MODELS_INFO = {
    "X1-Y2-Z1": {
        "home": [0.264589, -0.293903, 0.777] 
    },
    "X2-Y2-Z2": {
        "home": [0.277866, -0.724482, 0.777] 
    },
    "X1-Y3-Z2": {
        "home": [0.268053, -0.513924, 0.777]  
    },
    "X1-Y2-Z2": {
        "home": [0.429198, -0.293903, 0.777] 
    },
    "X1-Y2-Z2-CHAMFER": {
        "home": [0.592619, -0.293903, 0.777]  
    },
    "X1-Y4-Z2": {
        "home": [0.108812, -0.716057, 0.777] 
    },
    "X1-Y1-Z2": {
        "home": [0.088808, -0.295820, 0.777] 
    },
    "X1-Y2-Z2-TWINFILLET": {
        "home": [0.103547, -0.501132, 0.777] 
    },
    "X1-Y3-Z2-FILLET": {
        "home": [0.433739, -0.507130, 0.777]  
    },
    "X1-Y4-Z1": {
        "home": [0.589908, -0.501033, 0.777]  
    },
    "X2-Y2-Z2-FILLET": {
        "home": [0.442505, -0.727271, 0.777] 
    }
}

for model, model_info in MODELS_INFO.items():
    pass
    #MODELS_INFO[model]["home"] = model_info["home"] + np.array([0.0, 0.10, 0.0])

for model, info in MODELS_INFO.items():
    model_json_path = os.path.join(PKG_PATH, "..", "models", f"lego_{model}", "model.json")
    # make path absolute
    model_json_path = os.path.abspath(model_json_path)
    # check path exists
    if not os.path.exists(model_json_path):
        raise FileNotFoundError(f"Model file {model_json_path} not found")

    model_json = json.load(open(model_json_path, "r"))
    corners = np.array(model_json["corners"])

    size_x = (np.max(corners[:, 0]) - np.min(corners[:, 0]))
    size_y = (np.max(corners[:, 1]) - np.min(corners[:, 1]))
    size_z = (np.max(corners[:, 2]) - np.min(corners[:, 2]))

    #print(f"{model}: {size_x:.3f} x {size_y:.3f} x {size_z:.3f}")

    MODELS_INFO[model]["size"] = (size_x, size_y, size_z)

# Compensate for the interlocking height
INTERLOCKING_OFFSET = 0.019

SAFE_X = -0.40
SAFE_Y = -0.13
SURFACE_Z = 0.774

# Resting orientation of the end effector
DEFAULT_QUAT = PyQuaternion(axis=(0, 1, 0), angle=math.pi)
# Resting position of the end effector
DEFAULT_POS = (-0.1, -0.2, 1.2)

DEFAULT_PATH_TOLERANCE = control_msgs.msg.JointTolerance()
DEFAULT_PATH_TOLERANCE.name = "path_tolerance"
DEFAULT_PATH_TOLERANCE.velocity = 10

def get_gazebo_model_name(model_name, vision_model_pose):
    """
        Get the name of the model inside gazebo. It is needed for link attacher plugin.
    """
    models = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=None)
    epsilon = 0.05
    for gazebo_model_name, model_pose in zip(models.name, models.pose):
        if model_name not in gazebo_model_name:
            continue
        # Get everything inside a square of side epsilon centered in vision_model_pose
        ds = abs(model_pose.position.x - vision_model_pose.position.x) + abs(model_pose.position.y - vision_model_pose.position.y)
        if ds <= epsilon:
            return gazebo_model_name
    raise ValueError(f"Model {model_name} at position {vision_model_pose.position.x} {vision_model_pose.position.y} was not found!")


def get_model_name(gazebo_model_name):
    return gazebo_model_name.replace("lego_", "").split("_", maxsplit=1)[0]


def get_legos_pos(vision=False):
    #get legos position reading vision topic
    if vision:
        legos = rospy.wait_for_message("/lego_detections", ModelStates, timeout=None)
    else:
        models = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=None)
        legos = ModelStates()

        for name, pose in zip(models.name, models.pose):
            if "X" not in name:
                continue
            name = get_model_name(name)

            legos.name.append(name)
            legos.pose.append(pose)
    return [(lego_name, lego_pose) for lego_name, lego_pose in zip(legos.name, legos.pose)]


def straighten(model_pose, gazebo_model_name):
    x = model_pose.position.x
    y = model_pose.position.y
    z = model_pose.position.z
    model_quat = PyQuaternion(
        x=model_pose.orientation.x,
        y=model_pose.orientation.y,
        z=model_pose.orientation.z,
        w=model_pose.orientation.w)

    model_size = [0.5,0.5,0.5]

    """
        Calculate approach quaternion and target quaternion
    """

    facing_direction = get_axis_facing_camera(model_quat)
    approach_angle = get_approach_angle(model_quat, facing_direction)

    print(f"Lego is facing {facing_direction}")
    print(f"Angle of approaching measures {approach_angle:.2f} deg")

    # Calculate approach quat
    approach_quat = get_approach_quat(facing_direction, approach_angle)

    # Get above the object
    controller.move_to(x, y, target_quat=approach_quat)

    # Calculate target quat
    regrip_quat = DEFAULT_QUAT
    if facing_direction == (1, 0, 0) or facing_direction == (0, 1, 0):  # Side
        target_quat = DEFAULT_QUAT
        pitch_angle = -math.pi/2 + 0.2

        if abs(approach_angle) < math.pi/2:
            target_quat = target_quat * PyQuaternion(axis=(0, 0, 1), angle=math.pi/2)
        else:
            target_quat = target_quat * PyQuaternion(axis=(0, 0, 1), angle=-math.pi/2)
        target_quat = PyQuaternion(axis=(0, 1, 0), angle=pitch_angle) * target_quat

        if facing_direction == (0, 1, 0):
            regrip_quat = PyQuaternion(axis=(0, 0, 1), angle=math.pi/2) * regrip_quat

    elif facing_direction == (0, 0, -1):
        """
            Pre-positioning
        """
        controller.move_to(z=z, target_quat=approach_quat)
        close_gripper(gazebo_model_name, model_size[0])

        tmp_quat = PyQuaternion(axis=(0, 0, 1), angle=2*math.pi/6) * DEFAULT_QUAT
        controller.move_to(SAFE_X, SAFE_Y, z+0.05, target_quat=tmp_quat, z_raise=0.1)  # Move to safe position
        controller.move_to(z=z)
        open_gripper(gazebo_model_name)

        approach_quat = tmp_quat * PyQuaternion(axis=(1, 0, 0), angle=math.pi/2)

        target_quat = approach_quat * PyQuaternion(axis=(0, 0, 1), angle=-math.pi)  # Add a yaw rotation of 180 deg

        regrip_quat = tmp_quat * PyQuaternion(axis=(0, 0, 1), angle=math.pi)
    else:
        target_quat = DEFAULT_QUAT
        target_quat = target_quat * PyQuaternion(axis=(0, 0, 1), angle=-math.pi/2)

    """
        Grip the model
    """
    if facing_direction == (0, 0, 1) or facing_direction == (0, 0, -1):
        closure = model_size[0]
        z = SURFACE_Z + model_size[2] / 2
    elif facing_direction == (1, 0, 0):
        closure = model_size[1]
        z = SURFACE_Z + model_size[0] / 2
    elif facing_direction == (0, 1, 0):
        closure = model_size[0]
        z = SURFACE_Z + model_size[1] / 2
    controller.move_to(z=z, target_quat=approach_quat)
    close_gripper(gazebo_model_name, closure)

    """
        Straighten model if needed
    """
    if facing_direction != (0, 0, 1):
        z = SURFACE_Z + model_size[2]/2

        controller.move_to(z=z+0.05, target_quat=target_quat, z_raise=0.1)
        controller.move(dz=-0.05)
        open_gripper(gazebo_model_name)

        # Re grip the model
        controller.move_to(z=z, target_quat=regrip_quat, z_raise=0.1)
        close_gripper(gazebo_model_name, model_size[0])


def close_gripper(gazebo_model_name, closure=0):
    set_gripper(0.81-closure*10)
    rospy.sleep(0.5)
    # Create dynamic joint
    if gazebo_model_name is not None:
        req = AttachRequest()
        req.model_name_1 = gazebo_model_name
        req.link_name_1 = "link"
        req.model_name_2 = "robot"
        req.link_name_2 = "wrist_3_link"
        attach_srv.call(req)


def open_gripper(gazebo_model_name=None):
    set_gripper(0.0)

    # Destroy dynamic joint
    if gazebo_model_name is not None:
        req = AttachRequest()
        req.model_name_1 = gazebo_model_name
        req.link_name_1 = "link"
        req.model_name_2 = "robot"
        req.link_name_2 = "wrist_3_link"
        detach_srv.call(req)


def set_model_fixed(model_name):
    req = AttachRequest()
    req.model_name_1 = model_name
    req.link_name_1 = "link"
    req.model_name_2 = "ground_plane"
    req.link_name_2 = "link"
    attach_srv.call(req)

    req = SetStaticRequest()
    print("{} TO HOME".format(model_name))
    req.model_name = model_name
    req.link_name = "link"
    req.set_static = True

    setstatic_srv.call(req)


def get_approach_quat(facing_direction, approach_angle):
    quat = DEFAULT_QUAT
    if facing_direction == (0, 0, 1):
        pitch_angle = 0
        yaw_angle = 0
    elif facing_direction == (1, 0, 0) or facing_direction == (0, 1, 0):
        pitch_angle = + 0.2
        if abs(approach_angle) < math.pi/2:
            yaw_angle = math.pi/2
        else:
            yaw_angle = -math.pi/2
    elif facing_direction == (0, 0, -1):
        pitch_angle = 0
        yaw_angle = 0
    else:
        raise ValueError(f"Invalid model state {facing_direction}")

    quat = quat * PyQuaternion(axis=(0, 1, 0), angle=pitch_angle)
    quat = quat * PyQuaternion(axis=(0, 0, 1), angle=yaw_angle)
    quat = PyQuaternion(axis=(0, 0, 1), angle=approach_angle+math.pi/2) * quat

    return quat


def get_axis_facing_camera(quat):
    axis_x = np.array([1, 0, 0])
    axis_y = np.array([0, 1, 0])
    axis_z = np.array([0, 0, 1])
    new_axis_x = quat.rotate(axis_x)
    new_axis_y = quat.rotate(axis_y)
    new_axis_z = quat.rotate(axis_z)
    # get angle between new_axis and axis_z
    angle = np.arccos(np.clip(np.dot(new_axis_z, axis_z), -1.0, 1.0))
    # get if model is facing up, down or sideways
    if angle < np.pi / 3:
        return 0, 0, 1
    elif angle < np.pi / 3 * 2 * 1.2:
        if abs(new_axis_x[2]) > abs(new_axis_y[2]):
            return 1, 0, 0
        else:
            return 0, 1, 0
        #else:
        #    raise Exception(f"Invalid axis {new_axis_x}")
    else:
        return 0, 0, -1


def get_approach_angle(model_quat, facing_direction):#get gripper approach angle
    if facing_direction == (0, 0, 1):
        return model_quat.yaw_pitch_roll[0] - math.pi/2 #rotate gripper
    elif facing_direction == (1, 0, 0) or facing_direction == (0, 1, 0):
        axis_x = np.array([0, 1, 0])
        axis_y = np.array([-1, 0, 0])
        new_axis_z = model_quat.rotate(np.array([0, 0, 1])) #get z axis of lego
        # get angle between new_axis and axis_x
        dot = np.clip(np.dot(new_axis_z, axis_x), -1.0, 1.0) #sin angle between lego z axis and x axis in fixed frame
        det = np.clip(np.dot(new_axis_z, axis_y), -1.0, 1.0) #cos angle between lego z axis and x axis in fixed frame
        return math.atan2(det, dot) #get angle between lego z axis and x axis in fixed frame
    elif facing_direction == (0, 0, -1):
        return -(model_quat.yaw_pitch_roll[0] - math.pi/2) % math.pi - math.pi
    else:
        raise ValueError(f"Invalid model state {facing_direction}")


def set_gripper(value):
    goal = control_msgs.msg.GripperCommandGoal()
    goal.command.position = value  # From 0.0 to 0.8
    goal.command.max_effort = -1  # # Do not limit the effort
    action_gripper.send_goal_and_wait(goal, rospy.Duration(10))

    return action_gripper.get_result()


if __name__ == "__main__":
    print("Initializing node of kinematics")
    rospy.init_node("send_joints")

    controller = ArmController()

    # Create an action client for the gripper
    action_gripper = actionlib.SimpleActionClient(
        "/gripper_controller/gripper_cmd",
        control_msgs.msg.GripperCommandAction
    )
    print("Waiting for action of gripper controller")
    action_gripper.wait_for_server()

    setstatic_srv = rospy.ServiceProxy("/link_attacher_node/setstatic", SetStatic)
    attach_srv = rospy.ServiceProxy("/link_attacher_node/attach", Attach)
    detach_srv = rospy.ServiceProxy("/link_attacher_node/detach", Attach)
    setstatic_srv.wait_for_service()
    attach_srv.wait_for_service()
    detach_srv.wait_for_service()

    controller.move_to(*DEFAULT_POS, DEFAULT_QUAT)

    print("Waiting for detection of the models")
    rospy.sleep(0.5)
    msg = rospy.wait_for_message("/lego_detections", Float32MultiArray, timeout=None)
    print("Inside motion planning")
    d = list(msg.data)
    print("Data from message is", d)
    x, y, z = [0.264589, -0.293903, 0.777]
    grip_width = d[4]
    CURR_Z = 0  # Current end-effector z height.
    # Convert width in pixels to mm.
    # 0.07 is distance from end effector (CURR_Z) to camera.
    # 0.1 is approx degrees per pixel for the realsense.
    g_width = 2 * ((CURR_Z + 0.07)) * np.tan(0.1 * grip_width / 2.0 / 180.0 * np.pi) * 1000
    # Convert into motor positions.
    g = min((1 - (min(g_width, 70)/70)) * (6800-4000) + 4000, 5500)
    set_gripper(value=g)

    # Pose of the grasp (position only) in the camera frame.
    gp = geometry_msgs.msg.Pose()
    gp.position.x = d[2]
    gp.position.y = d[0]
    gp.position.z = d[1]
    gp.orientation.w = 1
    object_pose_camera = [gp.position.x, gp.position.y, gp.position.z, 0.0, 0.0, 0.0, 1]
    # Camera pose in robot frame, get these values from the tf transformation defined in lauch file 
    translation_camera_to_robot = [-0.44, -0.5, 1.58]
    rotation_camera_to_robot = [-0.5022940270013622, 0.49769222119830875, 0.5022972049227021, 0.4976953700048749]

    # Convert to transformation matrices
    transformation_camera_to_robot = tf_trans.concatenate_matrices(
        tf_trans.translation_matrix(translation_camera_to_robot),
        tf_trans.quaternion_matrix(rotation_camera_to_robot))

    object_pose_camera_matrix = tf_trans.concatenate_matrices(
        tf_trans.translation_matrix(object_pose_camera[:3]),
        tf_trans.quaternion_matrix(object_pose_camera[3:]))

    # Combine transformations
    object_pose_robot_matrix = np.dot(transformation_camera_to_robot, object_pose_camera_matrix)

    # Extract translation and rotation
    translation_object_in_robot = tf_trans.translation_from_matrix(object_pose_robot_matrix)
    rotation_object_in_robot = tf_trans.quaternion_from_matrix(object_pose_robot_matrix)
    #gp_base = convert_pose(gp, 'camera_link', 'base_link')
    #gp_base = convert_pose(gp, 'base_link','camera_link')
    q = tft.quaternion_from_euler(np.pi, 0, d[3])
    gp_base = geometry_msgs.msg.Pose()
    gp_base.position.x = translation_object_in_robot[0]
    gp_base.position.y = translation_object_in_robot[1]
    gp_base.position.z = translation_object_in_robot[2]
    gp_base.orientation.x = q[0]
    gp_base.orientation.y = q[1]
    gp_base.orientation.z = q[2]
    gp_base.orientation.w = q[3]

    publish_pose_as_transform(gp_base, 'base_link_base', 'G', 0.5)
    print("GP", gp_base)
    print("Q:", q)
    print(100*"-")
    print(100*"-")
    controller.move_to(x=gp_base.position.x,
                       y=gp_base.position.y, 
                       target_quat=PyQuaternion(
                                                x=gp_base.orientation.x,
                                                y=gp_base.orientation.y,
                                                z=gp_base.orientation.z,
                                                w=gp_base.orientation.w)
                        )
    print("Default POS", DEFAULT_POS)
    print("Default Quat", DEFAULT_QUAT)

    controller.move(dz=0.15)
    print("Moving to Default Position")
    controller.move_to(*DEFAULT_POS, DEFAULT_QUAT)
    open_gripper()
    rospy.sleep(0.4)