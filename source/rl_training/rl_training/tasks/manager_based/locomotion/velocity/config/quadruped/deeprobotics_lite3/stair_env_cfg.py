# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
from isaaclab.utils import configclass
from .rough_env_cfg import DeeproboticsLite3RoughEnvCfg

@configclass
class DeeproboticsLite3StairEnvCfg(DeeproboticsLite3RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ------------------------------ Terrains : majorité escaliers ------------------------------
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].proportion = 0.35
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.05
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.00
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion = 0.50
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 0.10

        # ------------------------------ Rewards : escalier focused ------------------------------
        self.rewards.lin_vel_z_l2.weight = -1.0            # corps peut monter
        self.rewards.ang_vel_xy_l2.weight = -0.03           # corps peut s'incliner
        self.rewards.flat_orientation_l2.weight = -2.5       # dos peut s'incliner
        self.rewards.feet_height.params["target_height"] = 0.16  # marches de 15.5cm
        self.rewards.feet_air_time.params["threshold"] = 0.55
        self.rewards.feet_gait.weight = 1.0                  # forcer trot diagonal
        self.rewards.joint_mirror.weight = -0.05             # symétrie souple
        self.rewards.feet_air_time_variance.weight = -0.0    # rythme libre

        # Anti-stumble : pénalise les pieds qui tapent les marches
        self.rewards.feet_stumble.weight = -0.2
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]

        # ------------------------------ Commands : vitesse prudente --------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)

        # ------------------------------ Disable zero rewards --------------------------------------
        if self.__class__.__name__ == "DeeproboticsLite3StairEnvCfg":
            self.disable_zero_weight_rewards()