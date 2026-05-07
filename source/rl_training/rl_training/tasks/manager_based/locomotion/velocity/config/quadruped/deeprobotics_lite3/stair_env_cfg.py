# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
from isaaclab.utils import configclass
from .rough_env_cfg import DeeproboticsLite3RoughEnvCfg

@configclass
class DeeproboticsLite3StairEnvCfg(DeeproboticsLite3RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ------------------------------ Terrains : Phase 1 stairs léger ------------------------------
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].proportion = 0.45
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.25
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.15
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion = 0.15
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 0.0

        # ------------------------------ Rewards : légèrement relâcher pour les escaliers --------------
        self.rewards.lin_vel_z_l2.weight = -1.0            # était -2.0 (corps peut monter)
        self.rewards.ang_vel_xy_l2.weight = -0.025          # était -0.05 (corps peut s'incliner)
        self.rewards.flat_orientation_l2.weight = -3.7       # était -5.0 (dos peut être incliné)
        self.rewards.feet_height.params["target_height"] = 0.17  # était 0.10 → CHANGÉ
        self.rewards.feet_air_time.params["threshold"] = 0.55    # était 0.5
        self.rewards.feet_gait.weight = 1.0            # était 0.5 (doubler)
        self.rewards.joint_mirror.weight = -0.15        # était -0.05 (tripler la pénalité)
        self.rewards.feet_air_time_variance.weight = -11.0   # était -8.0 → CHANGÉ

        # ------------------------------ Commands : vitesse prudente --------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)

        # ------------------------------ Disable zero rewards --------------------------------------
        if self.__class__.__name__ == "DeeproboticsLite3StairEnvCfg":
            self.disable_zero_weight_rewards()