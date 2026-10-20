#!/bin/bash
# Run inside the experiment environment:  bash collect_env.sh   (ZOOMEYE=path/to/ZoomEye to record its commit)
# Writes ENVIRONMENT.txt: GPU, driver, CUDA, Python packages, and the exact Hugging Face revisions used.
{
  echo "== date"; date -u
  echo "== GPU"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  echo "== CPU / OS"; lscpu | grep -E "Model name|^CPU\(s\)"; uname -sr; free -g | head -2
  echo "== python"; python --version
  python - <<'EOF'
import torch, transformers, numpy, importlib
print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
print("transformers", transformers.__version__, "numpy", numpy.__version__)
for m in ("accelerate", "qwen_vl_utils", "datasets", "huggingface_hub", "PIL", "pyarrow", "spacy"):
    try:
        print(m, importlib.import_module(m).__version__)
    except Exception as e:
        print(m, "n/a", type(e).__name__)
from huggingface_hub import scan_cache_dir
for r in scan_cache_dir().repos:
    if any(k in r.repo_id for k in ("Qwen2.5-VL", "InternVL3", "llava-onevision", "MME-RealWorld")):
        for rev in r.revisions:
            print("hf", r.repo_type, r.repo_id, rev.commit_hash)
EOF
  echo "== ZoomEye commit"; git -C "${ZOOMEYE:-ZoomEye}" rev-parse HEAD 2>/dev/null || echo "not cloned"
  echo "== pip freeze"; pip freeze
} > ENVIRONMENT.txt 2>&1
echo "wrote ENVIRONMENT.txt"
