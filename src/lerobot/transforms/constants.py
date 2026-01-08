from collections import defaultdict

from lerobot.utils.constants import OBS_STATE, ACTION, OBS_IMAGES, OBS_IMAGE
from .utils import make_bool_mask


MASK_MAPPING = {
    "piper": make_bool_mask(6, -1, 6, -1), 
    "arx_lift2": make_bool_mask(6, -1, 6, -1), 
    "split_aloha": make_bool_mask(6, -1, 6, -1), 
    "a2d": make_bool_mask(14, -2), 
    "genie1": make_bool_mask(14, -2), 
    "franka": make_bool_mask(7, -1), 
    "frankarobotiq": make_bool_mask(7, -1), 
    "aloha": make_bool_mask(6, -1, 6, -1), 
    "panda": make_bool_mask(7, ), 
}


FEATURE_MAPPING = defaultdict(
    lambda : {
        OBS_STATE: ["observation.state"],
        ACTION: ["action"],
    }, 
    a2d={
        OBS_STATE: [
            "observation.states.joint.position", 
            "observation.states.effector.position", 
        ], 
        ACTION: [
            "actions.joint.position", 
            "actions.effector.position", 
        ], 
    }, 
    genie1={
        OBS_STATE: [
            "states.left_joint.position", 
            "states.right_joint.position", 
            "states.left_gripper.position", 
            "states.right_gripper.position", 
        ], 
        ACTION: [
            "actions.left_joint.position", 
            "actions.right_joint.position", 
            "actions.left_gripper.position", 
            "actions.right_gripper.position", 
        ], 
    }, 
    arx_lift2={
        OBS_STATE: [
            "states.left_joint.position", 
            "states.left_gripper.position", 
            "states.right_joint.position", 
            "states.right_gripper.position", 
        ], 
        ACTION: [
            "actions.left_joint.position", 
            "actions.left_gripper.position", 
            "actions.right_joint.position", 
            "actions.right_gripper.position", 
        ], 
    }, 
    piper={
        OBS_STATE: [
            "states.left_joint.position", 
            "states.left_gripper.position", 
            "states.right_joint.position", 
            "states.right_gripper.position", 
        ], 
        ACTION: [
            "actions.left_joint.position", 
            "actions.left_gripper.position", 
            "actions.right_joint.position", 
            "actions.right_gripper.position", 
        ], 
    }, 
    r1lite={
        OBS_STATE: [
            'observation.state.left_arm', 
            'observation.state.right_arm', 
            'observation.state.left_gripper', 
            'observation.state.right_gripper',
        ], 
        ACTION: [
            "action.left_arm", 
            "action.right_arm",
            "action.left_gripper",
            "action.right_gripper",
        ], 
    },
    aloha={
        OBS_STATE: [
            'observation.state',
        ], 
        ACTION: [
            'action',
        ], 
    },
    franka={
        OBS_STATE: [
            "states.joint.position", 
            "states.gripper.position",
        ], 
        ACTION: [
            "actions.joint.position", 
            "actions.gripper.position", 
        ], 
    }, 
    panda={
        OBS_STATE: [
            "observation.state", 
        ], 
        ACTION: [
            "action", 
        ], 
    }, 
)


IMAGE_MAPPING = defaultdict(
    lambda : {
        "observation.image": f"{OBS_IMAGES}.image0", 
    }, 
    arx_lift2={
        "images.rgb.head": f"{OBS_IMAGES}.image0", 
        "images.rgb.hand_left": f"{OBS_IMAGES}.image1", 
        "images.rgb.hand_right": f"{OBS_IMAGES}.image2", 
    }, 
    piper={
        "images.rgb.head": f"{OBS_IMAGES}.image0", 
        "images.rgb.hand_left": f"{OBS_IMAGES}.image1", 
        "images.rgb.hand_right": f"{OBS_IMAGES}.image2", 
    },
    genie1={
        "images.rgb.head": f"{OBS_IMAGES}.image0", 
        "images.rgb.hand_left": f"{OBS_IMAGES}.image1", 
        "images.rgb.hand_right": f"{OBS_IMAGES}.image2", 
    }, 
    a2d={
        "observation.images.head": f"{OBS_IMAGES}.image0", 
        "observation.images.hand_left": f"{OBS_IMAGES}.image1", 
        "observation.images.hand_right": f"{OBS_IMAGES}.image2", 
    }, 
    # todo, make sure what the key names are for franka
    franka={
        "images.rgb.head": f"{OBS_IMAGES}.image0", 
        "images.rgb.hand": f"{OBS_IMAGES}.image1", 
    }, 
    r1lite={
        "observation.images.head_rgb": f"{OBS_IMAGES}.image0", 
        "observation.images.left_wrist_rgb": f"{OBS_IMAGES}.image1", 
        "observation.images.right_wrist_rgb": f"{OBS_IMAGES}.image2", 
    },

    aloha={
        "observation.images.cam_high": f"{OBS_IMAGES}.image0", 
        "observation.images.cam_left_wrist": f"{OBS_IMAGES}.image1", 
        "observation.images.cam_right_wrist": f"{OBS_IMAGES}.image2", 
    },
    panda={
        "observation.images.image": f"{OBS_IMAGES}.image0", 
        "observation.images.image2": f"{OBS_IMAGES}.image1", 
    },
)