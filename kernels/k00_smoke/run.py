"""k00 -- harness validation only. Proves push/poll/pull works before any science.

Fails fast and loudly: if the GPU is not visible we want to know in 30 seconds,
not at hour 4 of a grid.
"""
import json, os, platform, subprocess, sys, time

t0 = time.time()
out = "/kaggle/working"
os.makedirs(out, exist_ok=True)

info = {"python": sys.version.split()[0], "platform": platform.platform()}

import torch
info["torch"] = torch.__version__
info["cuda_available"] = torch.cuda.is_available()
assert torch.cuda.is_available(), "GPU not visible -- kernel misconfigured, abort"
info["gpu"] = torch.cuda.get_device_name(0)
info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
info["n_gpu"] = torch.cuda.device_count()
info["capability"] = "sm_%d%d" % torch.cuda.get_device_capability(0)
info["torch_arch_list"] = torch.cuda.get_arch_list()
# Kaggle's default P100 is sm_60, which current torch builds no longer ship kernels for.
# Catch it here, in 30 seconds, instead of at hour 4 of a grid.
assert info["capability"] in info["torch_arch_list"], (
    f"torch {torch.__version__} has no kernels for {info['gpu']} ({info['capability']}); "
    f"available: {info['torch_arch_list']}. Set machine_shape to T4x2.")

# measure real throughput so the grid's time budget is based on data, not guesswork
x = torch.randn(4096, 4096, device="cuda")
torch.cuda.synchronize(); t = time.time()
for _ in range(50):
    x = x @ x / 100
torch.cuda.synchronize()
info["matmul_4096_ms"] = round((time.time() - t) / 50 * 1000, 2)

# incremental write, per kernel design rule 2
with open(f"{out}/smoke.json", "w") as f:
    json.dump(info, f, indent=2)
info["elapsed_s"] = round(time.time() - t0, 1)
print(json.dumps(info, indent=2))
