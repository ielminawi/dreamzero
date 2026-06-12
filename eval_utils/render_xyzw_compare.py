"""Render xyzw vs xyzw_conj across several frames for side-by-side eyeballing.

Loads xyzw_compare.npz (export_xyzw_compare.py). For each frame, teleports the arms
to the xyzw solution and the xyzw_conj solution in turn (hands at recorded qpos_hand),
captures the robot cameras + an iso overview, and builds ONE sheet: a row per frame
  [ REAL aria | xyzw aria | conj aria | REAL oakd | xyzw oakd | conj oakd | xyzw iso | conj iso ]
You scan rows; the differences show up on whichever hand is away from 180deg.

Inside the Isaac container (docker/scripts/run_xyzw_compare.sh):
  ... -p eval_utils/render_xyzw_compare.py --headless --enable_cameras --device cpu \
      --h5 /app/20250826_111157.h5 --npz /app/xyzw_compare.npz --out-dir /output
"""
import argparse
import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--h5", type=str, default="/app/20250826_111157.h5")
parser.add_argument("--npz", type=str, required=True)
parser.add_argument("--out-dir", type=str, default="/output")
parser.add_argument("--warmup", type=int, default=30)
parser.add_argument("--settle", type=int, default=12)
parser.add_argument("--ov-w", type=int, default=1400)
parser.add_argument("--ov-h", type=int, default=1050)
parser.add_argument("--no-table", action="store_true",
                    help="remove the table and the objects resting on it (clean scene)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os, sys
sys.path.insert(0, "/app")
import cv2
import omni.kit.app
_em = omni.kit.app.get_app().get_extension_manager()
for _ext in ["isaacsim.asset.importer.urdf", "omni.importer.urdf", "omni.isaac.urdf"]:
    if _em.set_extension_enabled_immediate(_ext, True):
        break

import h5py
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
import sim_envs  # noqa: F401
from sim_envs.franka_orca_bimanual_cfg import (
    FrankaOrcaBimanualEnvCfg, LEFT_ARM_ORDER, RIGHT_ARM_ORDER, LEFT_HAND_ORDER, RIGHT_HAND_ORDER)
from sim_envs.franka_orca_bimanual_env import FrankaOrcaBimanualEnv

ARIA_DS = "observations/images/aria_rgb_cam/color"
OAKD_DS = "observations/images/oakd_front_view/color"
os.makedirs(args.out_dir, exist_ok=True)
def log(*a): print(" ".join(str(x) for x in a), flush=True)

def label(img, text, color=(255,255,255), scale=0.6):
    out=np.ascontiguousarray(img); h=max(22,int(26*scale/0.6))
    cv2.rectangle(out,(0,0),(out.shape[1],h),(0,0,0),-1)
    cv2.putText(out,text,(6,int(h*0.74)),cv2.FONT_HERSHEY_SIMPLEX,scale,color,2,cv2.LINE_AA)
    return out
def fit(img,w,h): return cv2.resize(img[...,:3].astype(np.uint8),(w,h),interpolation=cv2.INTER_AREA)

ISO=("iso",(1.15,-1.15,0.95),(0.28,0.0,0.12),14.0)
FRONT=("front",(1.42,0.0,0.70),(0.05,0.0,0.25),8.0)   # straight-on +X, wide (wall at x=1.5)

def main():
    npz=np.load(args.npz,allow_pickle=True)
    frames=[int(x) for x in npz["frames"]]
    ikL=npz["ik_left"].astype(np.float32); ikR=npz["ik_right"].astype(np.float32)  # (F,2,7)
    clab=[str(x) for x in npz["cand_labels"]]

    cfg=FrankaOrcaBimanualEnvCfg(); cfg.sim.device=args.device
    if args.no_table:
        for asset in ["table","grocery_bag","white_container","blue_tube","round_item"]:
            if hasattr(cfg.scene,asset): setattr(cfg.scene,asset,None)
        log("removed table + objects from the scene")
    for name,eye,tgt,foc in (ISO,FRONT):
        setattr(cfg.scene,f"ov_{name}",CameraCfg(prim_path=f"/World/OverviewCams/{name}",update_period=0.0,
            height=args.ov_h,width=args.ov_w,
            spawn=sim_utils.PinholeCameraCfg(focal_length=foc,horizontal_aperture=20.955),
            offset=CameraCfg.OffsetCfg(pos=eye,rot=(1.0,0.0,0.0,0.0),convention="world")))
    env=FrankaOrcaBimanualEnv(cfg); env.reset(); env.reset()
    for name,eye,tgt,foc in (ISO,FRONT):
        env.scene[f"ov_{name}"].set_world_poses_from_view(
            torch.tensor([eye],dtype=torch.float32,device=env.device),
            torch.tensor([tgt],dtype=torch.float32,device=env.device))

    idsL,_=env.scene["left_arm_hand"].find_joints(list(LEFT_ARM_ORDER)+list(LEFT_HAND_ORDER),preserve_order=True)
    idsR,_=env.scene["right_arm_hand"].find_joints(list(RIGHT_ARM_ORDER)+list(RIGHT_HAND_ORDER),preserve_order=True)

    f=h5py.File(args.h5,"r")
    first=True; rows=[]; frows=[]; RH=380
    for fi,Fr in enumerate(frames):
        q_hl=f["observations/qpos_hand_left"][Fr].astype(np.float32)
        q_hr=f["observations/qpos_hand_right"][Fr].astype(np.float32)
        real_aria=f[ARIA_DS][Fr].astype(np.uint8); real_oakd=f[OAKD_DS][Fr].astype(np.uint8)
        cap={}
        for ci in range(2):
            valsL=np.concatenate([ikL[fi,ci],q_hl]).astype(np.float32)
            valsR=np.concatenate([ikR[fi,ci],q_hr]).astype(np.float32)
            for skey,ids,vals in [("left_arm_hand",idsL,valsL),("right_arm_hand",idsR,valsR)]:
                art=env.scene[skey]; pos=torch.tensor(vals,device=env.device).unsqueeze(0)
                art.write_joint_state_to_sim(pos,torch.zeros_like(pos),joint_ids=ids)
                art.set_joint_position_target(pos,joint_ids=ids)
            env.scene.write_data_to_sim()
            a=torch.zeros((1,48),dtype=torch.float32,device=env.device)
            a[0,0:7]=torch.tensor(ikL[fi,ci],device=env.device); a[0,7:14]=torch.tensor(ikR[fi,ci],device=env.device)
            a[0,14:31]=torch.tensor(q_hl,device=env.device); a[0,31:48]=torch.tensor(q_hr,device=env.device)
            for _ in range(args.warmup if first else args.settle): env.step(a)
            first=False
            cap[ci]={
                "aria":env.scene["aria_rgb_cam"].data.output["rgb"][0,...,:3].cpu().numpy().astype(np.uint8),
                "oakd":env.scene["oakd_front_view"].data.output["rgb"][0,...,:3].cpu().numpy().astype(np.uint8),
                "iso":env.scene["ov_iso"].data.output["rgb"][0,...,:3].cpu().numpy().astype(np.uint8),
                "front":env.scene["ov_front"].data.output["rgb"][0,...,:3].cpu().numpy().astype(np.uint8)}
        row=np.concatenate([
            label(fit(real_aria,420,RH),f"REAL aria t={Fr}",(120,255,120)),
            label(fit(cap[0]["aria"],420,RH),"xyzw aria",(120,200,255)),
            label(fit(cap[1]["aria"],420,RH),"conj aria",(255,170,120)),
            label(fit(real_oakd,500,RH),f"REAL oakd t={Fr}",(120,255,120)),
            label(fit(cap[0]["oakd"],500,RH),"xyzw oakd",(120,200,255)),
            label(fit(cap[1]["oakd"],500,RH),"conj oakd",(255,170,120)),
            label(fit(cap[0]["iso"],460,RH),"xyzw iso",(120,200,255)),
            label(fit(cap[1]["iso"],460,RH),"conj iso",(255,170,120)),
        ],axis=1)
        rows.append(row)
        # dedicated FULL FRONTAL sheet row: big [xyzw front | conj front] tiles
        FH=620
        frow=np.concatenate([
            label(fit(cap[0]["front"],820,FH),f"xyzw FRONT t={Fr}",(120,200,255)),
            label(fit(cap[1]["front"],820,FH),f"conj FRONT t={Fr}",(255,170,120)),
        ],axis=1)
        frows.append(frow)
        log(f"  captured t={Fr} ({fi+1}/{len(frames)})")
    f.close()
    sheet=np.concatenate(rows,axis=0)
    outp=os.path.join(args.out_dir,"xyzw_vs_conj_sheet.png")
    cv2.imwrite(outp,cv2.cvtColor(sheet,cv2.COLOR_RGB2BGR))
    fsheet=np.concatenate(frows,axis=0)
    foutp=os.path.join(args.out_dir,"xyzw_vs_conj_front.png")
    cv2.imwrite(foutp,cv2.cvtColor(fsheet,cv2.COLOR_RGB2BGR))
    log("wrote",outp); log("wrote",foutp); log("DONE"); env.close(); simulation_app.close()

if __name__=="__main__":
    try: main()
    except Exception:
        import traceback; log("\n*** EXCEPTION ***"); log(traceback.format_exc()); os._exit(1)
