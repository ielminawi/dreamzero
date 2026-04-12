# Running Isaac Sim on Euler Cluster (RTX 4090)

## One-Time Setup

Create persistent cache directories on scratch (only needed once):

```bash
mkdir -p /cluster/scratch/ielminawi/isaacsim-cache/{kit-cache,kit-logs,isaaclabtemp,mpl,ov-cache,ov-data,ov-config}
mkdir -p /cluster/scratch/ielminawi/dreamzero/output
```

---

## Every Session

### 1. Request a Job Allocation

From the login node, request an interactive session with a GPU and enough memory:

```bash
srun --gpus=rtx_4090:1 --cpus-per-task=8 --mem-per-cpu=16G --tmp=100G --time=02:00:00 --pty bash
```

- `--cpus-per-task=8 --mem-per-cpu=16G` → 128GB total RAM (required for shader compilation)
- `--tmp=100G` → 100GB local scratch
- Adjust `--time` as needed (max depends on cluster policy)

### 2. Enable Internet Access

Isaac Sim needs to sync its extension registry on first launch:

```bash
module load eth_proxy
```

### 3. Launch the Apptainer Container

```bash
apptainer shell --nv --writable-tmpfs \
  --bind /cluster/scratch/ielminawi/dreamzero:/app \
  --bind /cluster/scratch/ielminawi/dreamzero/output:/output \
  --bind /cluster/scratch/ielminawi/isaacsim-cache/kit-cache:/isaac-sim/kit/cache \
  --bind /cluster/scratch/ielminawi/isaacsim-cache/kit-logs:/isaac-sim/kit/logs \
  --bind /cluster/scratch/ielminawi/isaacsim-cache/isaaclabtemp:/tmp/IsaacLab \
  --env MPLCONFIGDIR=/cluster/scratch/ielminawi/isaacsim-cache/mpl \
  --env XDG_CACHE_HOME=/cluster/scratch/ielminawi/isaacsim-cache/ov-cache \
  --env XDG_DATA_HOME=/cluster/scratch/ielminawi/isaacsim-cache/ov-data \
  --env XDG_CONFIG_HOME=/cluster/scratch/ielminawi/isaacsim-cache/ov-config \
  /cluster/scratch/ielminawi/isaaclab-bimanual-sandbox
```

**Why each bind:**
| Bind | Reason |
|------|--------|
| `dreamzero:/app` | Your codebase inside the container |
| `output:/output` | Where rendered images are saved |
| `kit-cache:/isaac-sim/kit/cache` | Persists shader/extension cache across sessions |
| `kit-logs:/isaac-sim/kit/logs` | Prevents verbose logs from filling tmpfs |
| `isaaclabtemp:/tmp/IsaacLab` | URDF→USD converter needs a writable temp dir |
| `XDG_*` env vars | Redirect Omniverse caches to scratch |

---

## Running the Test Scene

### Headless (no streaming)
```bash
/opt/IsaacLab/isaaclab.sh -p /app/eval_utils/test_scene.py --headless
```

### Headless + Save Camera Images
```bash
/opt/IsaacLab/isaaclab.sh -p /app/eval_utils/test_scene.py --headless --save-images
```
Images are saved to `/cluster/scratch/ielminawi/dreamzero/output/`:
- `aria_rgb_cam.png`
- `oakd_front_view.png`
- `composite.png`
- `debug_top_down.png`, `debug_front.png`, `debug_left_side.png`, `debug_perspective.png`

### Headless + WebRTC Livestream (run forever)
```bash
/opt/IsaacLab/isaaclab.sh -p /app/eval_utils/test_scene.py --headless --livestream 2 --num-steps 0
```

---

## Connecting the WebRTC Streaming Client

Isaac Sim binds its WebRTC streaming server on port **8011**.

### 1. Find your compute node hostname
Check the shell prompt — it shows the node name, e.g. `ielminawi@eu-g6-075`.

### 2. Set up an SSH tunnel on your Mac
```bash
ssh -L 8011:eu-g6-075:8011 ielminawi@euler.ethz.ch
```
Replace `eu-g6-075` with your actual compute node. Keep this terminal open.

### 3. Connect with the Isaac Sim WebRTC Streaming Client
- Open the **Isaac Sim WebRTC Streaming Client** app
- Server: `127.0.0.1`
- Click **Connect**

---

## Notes

- **First launch is slow** (~10-15 min) due to shader cache compilation. Subsequent launches are fast because caches persist on scratch.
- **`module load eth_proxy`** is required every session for the extension registry sync. After the first run the cache is warm and it resolves quickly.
- **Job time limit**: if the job is killed with `DUE TO TIME LIMIT`, just request a new allocation with a longer `--time`.
- The warnings about GLFW, NGX/DLSS, and `rendering_modes` in the Isaac Sim output are all benign in headless mode.
