# Isaac Sim + IsaacLab + Bimanual Franka Orca Environment
ARG ISAAC_SIM_VERSION=4.5.0
FROM nvcr.io/nvidia/isaac-sim:${ISAAC_SIM_VERSION}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install git (not included in Isaac Sim base image)
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Install IsaacLab
WORKDIR /opt
RUN git clone https://github.com/isaac-sim/IsaacLab.git && \
    cd IsaacLab && \
    ./isaaclab.sh --install

# Install additional Python dependencies for eval client
RUN /opt/IsaacLab/isaaclab.sh -p -m pip install --no-cache-dir \
    websockets \
    msgpack \
    msgpack-numpy \
    openpi-client==0.1.1 \
    tyro \
    mediapy \
    tqdm

WORKDIR /app

# Copy sim environment, eval code, configs, and URDF assets (including meshes)
COPY sim_envs/ /app/sim_envs/
COPY eval_utils/ /app/eval_utils/
COPY configs/ /app/configs/

# Enable Omniverse Streaming for remote visualization
ENV LIVESTREAM_ENABLED=true
ENV OMNI_KIT_ALLOW_ROOT=1

# Ports: 8211 = streaming signaling, 49100 = streaming data
EXPOSE 8211 49100

ENTRYPOINT ["/opt/IsaacLab/isaaclab.sh", "-p"]
CMD ["eval_utils/run_sim_eval_bimanual.py", \
     "--headless", \
     "--enable", "omni.kit.livestream.native"]
