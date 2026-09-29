#!/usr/bin/env bash
# Re-fetch the datasets/code the reproductions need (not stored in git).
# Run from the repo root:  bash setup_datasets.sh
set -e

# res-cnn3_paper: ELPV EL cells (Deitsch et al. 2019)
git clone --depth 1 https://github.com/zae-bayern/elpv-dataset.git \
    res-cnn3_paper/elpv-dataset

# udensenet_paper: RaptorMaps InfraredSolarModules (20k IR imgs, 12 classes)
git clone --depth 1 https://github.com/RaptorMaps/InfraredSolarModules.git \
    udensenet_paper/InfraredSolarModules
python -c "import zipfile; zipfile.ZipFile('udensenet_paper/InfraredSolarModules/2020-02-14_InfraredSolarModules.zip').extractall('udensenet_paper/InfraredSolarModules')"

# elseg_paper: UCF-EL-Defect (code + 17,064 EL imgs). Large repo; HTTP/1.1 avoids
# the Windows fetch-pack disconnect.
git -c http.version=HTTP/1.1 -c http.postBuffer=1048576000 clone --depth 1 \
    https://github.com/ucf-photovoltaics/UCF-EL-Defect.git \
    elseg_paper/UCF-EL-Defect

echo "datasets ready"
