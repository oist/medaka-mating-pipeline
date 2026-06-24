#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# install_detectron2_unit.sh  — install Detectron2 as an OIST saion unit module
#
# Mirrors the convention of ~/scripts/install/misc/install_sam3.sh (system python
# module -> venv in the unit tree -> torch CUDA wheels -> from-source package ->
# auto-generated #%Module1.0 modulefile), with the group/permission handling of
# install_sleap_nn_uv_shared.sh.
#
# Why a venv + torch wheels (not the system cuda module): saion's cuda modules
# top out at cuda/11.3, too old for current torch. The torch CUDA *runtime* comes
# bundled in the cu12x wheels; Detectron2's optional `_C` extension compiles
# CPU-only with gcc (no system nvcc needed) and standard Mask R-CNN inference
# still runs fully on GPU via torch/torchvision. If a model needs Detectron2's
# custom CUDA ops (deformable conv etc.), set CUDA_HOME to a matching nvcc and
# FORCE_CUDA=1.
#
# Produces:
#   ${UNIT_ROOT}/detectron2/${D2_VER}/venv/            (the environment)
#   ${UNIT_ROOT}/.modulefiles/detectron2/${D2_VER}     (the modulefile)
#
# Usage:
#   bash install_detectron2_unit.sh                    # defaults below
#   D2_VER=0.6-cu121 UNIT=StephensU bash install_detectron2_unit.sh
#
# Use it afterwards (and with the dual-line HPC wrapper):
#   module use ${UNIT_ROOT}/.modulefiles && module load detectron2/${D2_VER}
#   D2_SETUP="module use ${UNIT_ROOT}/.modulefiles && module load detectron2/${D2_VER}" \
#       bash scripts/hpc/submit_detectron2_experiment.sh 20251225_K_1_2_FCMC SKIP_VIEWS=top_cam
# =============================================================================

# ---- Config (env-overridable) ----------------------------------------------
UNIT="${UNIT:-StephensU}"                       # StephensU (medaka) | ReiterU (vision tooling)
GROUP="${GROUP:-stephensuni}"                    # stephensuni | reiteruni — must match UNIT
UNIT_ROOT="${UNIT_ROOT:-/apps/unit/${UNIT}}"
PKG="${PKG:-detectron2}"                          # module/dir name (e.g. detectron.2 to match the existing ReiterU module)
D2_VER="${D2_VER:-0.6-cu121}"                    # module version label (free-form)
PREFIX="${PREFIX:-${UNIT_ROOT}/${PKG}/${D2_VER}}"
MODULE_ROOT="${MODULE_ROOT:-${UNIT_ROOT}/.modulefiles}"
MODFILE="${MODULE_ROOT}/${PKG}/${D2_VER}"

PY_MODULE="${PY_MODULE:-python/3.10.10}"          # saion module providing the base python
GCC_MODULE="${GCC_MODULE:-gcc/11.2.1}"            # C++ compiler for the detectron2 ext build
CUDA_WHEEL_INDEX="${CUDA_WHEEL_INDEX:-https://download.pytorch.org/whl/cu121}"
D2_GIT="${D2_GIT:-https://github.com/facebookresearch/detectron2.git}"
D2_GIT_REF="${D2_GIT_REF:-}"                       # optional commit/tag to pin (e.g. v0.6)
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;8.0}"  # V100=7.0, A100=8.0
FORCE_CUDA="${FORCE_CUDA:-0}"                      # 1 = require a system nvcc (CUDA_HOME) for the ext
# Extra runtime deps the medaka Line-B pipeline (13_run_detectron2_segmentation) needs:
PIPELINE_DEPS="${PIPELINE_DEPS:-opencv-python-headless h5py pyyaml numpy}"

# ---- GPU-node re-exec (so torch.cuda.is_available() is true for the smoke test) ----
RUN_ON_GPU="${RUN_ON_GPU:-1}"
SRUN_PARTITION="${SRUN_PARTITION:-gpu}"           # gpu = V100, lower queue pressure
SRUN_CPUS="${SRUN_CPUS:-8}"; SRUN_MEM="${SRUN_MEM:-32G}"; SRUN_GRES="${SRUN_GRES:-gpu:1}"
SRUN_TIME="${SRUN_TIME:-02:00:00}"

log() { echo "[$(date +'%F %T')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

if [[ "$RUN_ON_GPU" == "1" && -z "${SLURM_JOB_ID:-}" ]]; then
  SCRIPT_PATH="$(readlink -f "$0")"
  log "Re-running on partition=${SRUN_PARTITION} (--gres=${SRUN_GRES}) for a GPU-visible build..."
  exec srun -p "$SRUN_PARTITION" -c "$SRUN_CPUS" --mem="$SRUN_MEM" --gres="$SRUN_GRES" \
       --time="$SRUN_TIME" --pty bash -lc "RUN_ON_GPU=0 \
       UNIT='${UNIT}' GROUP='${GROUP}' UNIT_ROOT='${UNIT_ROOT}' D2_VER='${D2_VER}' \
       PREFIX='${PREFIX}' MODULE_ROOT='${MODULE_ROOT}' PY_MODULE='${PY_MODULE}' \
       GCC_MODULE='${GCC_MODULE}' CUDA_WHEEL_INDEX='${CUDA_WHEEL_INDEX}' D2_GIT='${D2_GIT}' \
       D2_GIT_REF='${D2_GIT_REF}' TORCH_CUDA_ARCH_LIST='${TORCH_CUDA_ARCH_LIST}' \
       FORCE_CUDA='${FORCE_CUDA}' PIPELINE_DEPS='${PIPELINE_DEPS}' \
       bash '${SCRIPT_PATH}'"
fi

log "UNIT=${UNIT}  GROUP=${GROUP}"
log "PREFIX=${PREFIX}"
log "MODFILE=${MODFILE}"
log "PY_MODULE=${PY_MODULE}  GCC_MODULE=${GCC_MODULE}  wheels=${CUDA_WHEEL_INDEX}"
log "node=$(hostname)"

umask 002
if [[ -e "$PREFIX" ]]; then
  if [[ "${REPLACE:-0}" == "1" ]]; then
    log "REPLACE=1: removing existing ${PREFIX}"; rm -rf "$PREFIX"
  else
    die "Target exists: $PREFIX (set REPLACE=1 to overwrite, or bump D2_VER)"
  fi
fi
mkdir -p "$PREFIX" "$(dirname "$MODFILE")"

# ---- Toolchain ----
module purge
module load "$PY_MODULE" || die "failed to load $PY_MODULE"
module load "$GCC_MODULE" || log "WARN: could not load $GCC_MODULE (will use system gcc)"
command -v python >/dev/null || die "no python after module load"
log "python: $(python --version 2>&1)  gcc: $(gcc --version 2>&1 | head -1)"

# ---- venv ----
log "creating venv at ${PREFIX}/venv"
python -m venv "${PREFIX}/venv"
source "${PREFIX}/venv/bin/activate"
pip install --upgrade pip setuptools wheel ninja

# ---- torch (CUDA wheels — runtime bundled) ----
log "installing torch/torchvision from ${CUDA_WHEEL_INDEX}"
pip install torch torchvision --index-url "${CUDA_WHEEL_INDEX}"
python -c 'import torch; print("torch", torch.__version__, "cuda_built", torch.version.cuda)'

# ---- pipeline runtime deps (force WHEELS) ----
# The compute node's system HDF5 is ancient (1.8.12) and leaks via HDF5_* env, which
# makes h5py attempt a source build that fails. The manylinux wheels bundle their own
# HDF5/libs, so pin to binary-only and clear the leaking env.
log "installing pipeline deps (wheels only): ${PIPELINE_DEPS}"
unset HDF5_DIR HDF5_VERSION HDF5_PLUGIN_PATH 2>/dev/null || true
# shellcheck disable=SC2086
pip install --only-binary=:all: ${PIPELINE_DEPS}

# ---- Detectron2 from source ----
export TORCH_CUDA_ARCH_LIST FORCE_CUDA
if [[ "$FORCE_CUDA" == "1" ]]; then
  [[ -z "${CUDA_HOME:-}" ]] && \
    die "FORCE_CUDA=1 but CUDA_HOME unset — point it at an nvcc matching the torch CUDA major version"
else
  unset CUDA_HOME 2>/dev/null || true   # build CPU-only _C ext; GPU inference still via torch/torchvision
fi
D2_SPEC="git+${D2_GIT}"
[[ -n "$D2_GIT_REF" ]] && D2_SPEC="${D2_SPEC}@${D2_GIT_REF}"
log "building detectron2 from ${D2_SPEC}  (TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}, FORCE_CUDA=${FORCE_CUDA})"
pip install --no-build-isolation --prefer-binary "${D2_SPEC}"

# ---- smoke test ----
log "smoke test"
python - <<'PY'
import torch, detectron2, cv2, h5py, yaml, numpy
from detectron2.engine import DefaultPredictor   # noqa: F401  (import path used by 13_run_detectron2_segmentation)
print("detectron2", detectron2.__version__, "| torch", torch.__version__,
      "| cuda_available", torch.cuda.is_available(),
      "| cv2", cv2.__version__, "| h5py", h5py.__version__)
PY
deactivate

# ---- permissions (group-shared, setgid like the sleap installer) ----
chgrp -R "$GROUP" "$PREFIX" 2>/dev/null || true
chmod -R g+rX "$PREFIX"
find "$PREFIX" -type d -exec chmod g+s {} \; 2>/dev/null || true

# ---- modulefile (#%Module1.0, same shape as detectron.2/0.6 + SAM2) ----
log "writing modulefile ${MODFILE}"
cat > "$MODFILE" <<EOF
#%Module1.0#####################################################################
##
## Auto-generated ${PKG} ${D2_VER} module
##   built: $(date +'%F %T')   torch wheels: ${CUDA_WHEEL_INDEX}
##   arches: ${TORCH_CUDA_ARCH_LIST}   from: ${D2_GIT}${D2_GIT_REF:+@${D2_GIT_REF}}
##
proc ModulesHelp { } {
    puts stderr "Detectron2 ${D2_VER} (venv; torch CUDA wheels) for the medaka dual-line Line B."
}
module-whatis "Detectron2 ${D2_VER} — FAIR instance segmentation (venv at ${PREFIX}/venv)"

set root ${PREFIX}

# base python module the venv was built against
module load ${PY_MODULE}

setenv DETECTRON2_ROOT \$root
setenv VIRTUAL_ENV     \$root/venv
prepend-path PATH      \$root/venv/bin

# HPC hygiene
setenv PYTHONDONTWRITEBYTECODE 1
setenv PYTHONPYCACHEPREFIX /tmp/\$env(USER)/pycache
EOF
chgrp "$GROUP" "$MODFILE" 2>/dev/null || true
chmod 664 "$MODFILE"

log "DONE."
echo
echo "  module use ${MODULE_ROOT} && module load ${PKG}/${D2_VER}"
echo "  python -c 'import detectron2, torch; print(detectron2.__version__, torch.cuda.is_available())'"
echo
echo "Wire into the dual-line HPC wrapper:"
echo "  D2_SETUP=\"module use ${MODULE_ROOT} && module load ${PKG}/${D2_VER}\" \\"
echo "      bash scripts/hpc/submit_detectron2_experiment.sh 20251225_K_1_2_FCMC SKIP_VIEWS=top_cam"
echo "  # or directly: D2_PYTHON=${PREFIX}/venv/bin/python"
