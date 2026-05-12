from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.stage import open_stage
import omni.usd


SCENE_USD = "/home/ielminawi/setup/dual_franka_orca_physics.usda"


open_stage(SCENE_USD)
stage = omni.usd.get_context().get_stage()

if stage is None:
    raise RuntimeError(f"Could not open USD stage: {SCENE_USD}")

print(f"Opened {SCENE_USD}")
print("Press Play in Isaac Sim to test the ORCA physics scene.")

while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
