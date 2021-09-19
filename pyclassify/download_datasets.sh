#!/usr/bin/env bash
#
# Grabs datasets required to run these examples.  We don't store them in this
# git repo since they're big and binary.
#
# Syntax:
#
#   ./download_datasets.sh           symlink datasets if found, else download
#   ./download_datasets.sh remote    forces downloading datasets from TI
#   ./download_datasets.sh clean     removes datasets from this directory
#

declare -A rel_path

TIDL_LOCAL_BASE="/usr/share/ti/examples/tidl"
TIDL_REPO_BASE="https://git.ti.com/cgit/tidl/tidl-api/plain/examples"

rel_path[cat-pet-animal-domestic-104827.jpeg]="test/testvecs/input/objects"
rel_path[imagenet_objects.json]="mobilenet_subgraph"
rel_path[tidl_net_imagenet_jacintonet11v2.bin]="test/testvecs/config/tidl_models"
rel_path[tidl_param_imagenet_jacintonet11v2.bin]="test/testvecs/config/tidl_models"

for dest_file in "${!rel_path[@]}"; do
    if [ ! -f "$dest_file" ]; then
        local_version="${TIDL_LOCAL_BASE}/${rel_path[$dest_file]}/$dest_file"
        remote_version="${TIDL_REPO_BASE}/${rel_path[$dest_file]}/$dest_file"
        if [ -f "$local_version" -a "z$1" != "zremote" ]; then
            ln -sv "$local_version" .
        else
            curl -O "$remote_version"
        fi
    elif [[ $1 == "clean" ]]; then
        rm -v "$dest_file"
    else
        echo "$dest_file already exists"
    fi
done
