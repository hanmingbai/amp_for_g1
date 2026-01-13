from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from .base.legged_robot import LeggedRobot
from legged_gym.utils.task_registry import task_registry
from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from legged_gym.envs.g1.g1_env import G1Robot
from legged_gym.envs.h1.h1_config import H1RoughCfg, H1RoughCfgPPO
from legged_gym.envs.h1.h1_env import H1Robot
from .base.legged_robot_amp import LeggedRobotAmp

from legged_gym.envs.g1.g1_amp_12dof_config import G1AMP12DOFCfg, G1AMP12DOFCfgPPO
from legged_gym.envs.g1.g1_amp_23dof_config import G1AMP23DOFCfg, G1AMP23DOFCfgPPO
from legged_gym.envs.g1.g1_amp_config import G1AMPCfg, G1AMPCfgPPO


# g1 task registry
task_registry.register( "g1_12dof", G1Robot, G1RoughCfg(), G1RoughCfgPPO())

task_registry.register( "g1_amp_12dof", LeggedRobotAmp, G1AMP12DOFCfg(), G1AMP12DOFCfgPPO())

task_registry.register( "g1_amp_23dof", LeggedRobotAmp, G1AMP23DOFCfg(), G1AMP23DOFCfgPPO())

task_registry.register( "g1_amp", LeggedRobotAmp, G1AMPCfg(), G1AMPCfgPPO())

