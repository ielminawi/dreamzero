"""Check IsaacLab API availability."""
from isaaclab.sensors import CameraCfg
print("CameraCfg attrs:", [a for a in dir(CameraCfg) if not a.startswith("_")])
from isaaclab.actuators import ImplicitActuatorCfg
print("ImplicitActuatorCfg OK")
import isaaclab.sim as sim_utils
print("sim_utils has UrdfFileCfg:", hasattr(sim_utils, "UrdfFileCfg"))
print("sim_utils has UsdFileCfg:", hasattr(sim_utils, "UsdFileCfg"))
from isaaclab.envs import ManagerBasedEnvCfg
print("ManagerBasedEnvCfg attrs:", [a for a in dir(ManagerBasedEnvCfg) if not a.startswith("_")])
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
print("ObservationGroupCfg attrs:", [a for a in dir(ObservationGroupCfg) if not a.startswith("_")])
