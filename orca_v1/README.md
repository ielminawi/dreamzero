# ORCA v1 model

directory structure organized as advised in /source/isaaclab_assets/docs/README.md

There are a lot of gotchas during the conversion- generated with:

1. make sure base body is named "worldBody" (see [this issue](https://github.com/isaac-sim/IsaacLab/discussions/2171))
2. run converter
    ```bash
    python scripts/tools/convert_mjcf.py \
        source/isaaclab_assets/data/Robots/SRL/orca_v1/mjcf/orca_v1.xml \
        source/isaaclab_assets/data/Robots/SRL/orca_v1/orca_v1.usd \
        --fix-base --import-sites --make-instanceable
    ```
3. uncheck "Instanceable" for the Xform, and check it for the geoms (see [this issue](https://github.com/isaac-sim/IsaacLab/discussions/2113#discussioncomment-12690286)), and save using "Collect and Save As..." (see [this issue](https://github.com/isaac-sim/IsaacLab/discussions/2514))

## references
- https://isaac-sim.github.io/IsaacLab/main/source/how-to/import_new_asset.html
