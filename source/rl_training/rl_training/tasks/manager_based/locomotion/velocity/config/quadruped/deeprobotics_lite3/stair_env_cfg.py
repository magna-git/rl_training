# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
from isaaclab.utils import configclass
from rl_training.tasks.manager_based.locomotion.velocity.config.quadruped.deeprobotics_lite3.rough_env_cfg import DeeproboticsLite3RoughEnvCfg

@configclass
class DeeproboticsLite3StairEnvCfg(DeeproboticsLite3RoughEnvCfg):

    def __post_init__(self):
        super().__post_init__()

        # 1. TERRAINS
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion = 0.4
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 0.3
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].proportion = 0.2
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.05
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.05
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].proportion = 0.0

        # 2. REWARDS
        self.rewards.lin_vel_z_l2.weight = -0.5
        self.rewards.ang_vel_xy_l2.weight = -0.01
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.joint_power.weight = -5e-6
        self.rewards.feet_height.params["target_height"] = 0.15
        self.rewards.feet_air_time.params["threshold"] = 0.65
        self.rewards.feet_gait.weight = 0.1
        self.rewards.feet_air_time_variance.weight = -4.0

        # 3. COMMANDES
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)