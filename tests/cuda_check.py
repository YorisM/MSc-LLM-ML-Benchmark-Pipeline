# cuda_check.py
import time, torch, platform

print("PyTorch           :", torch.__version__)
print("Python            :", platform.python_version())
print("CUDA available    :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU               :", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
    print("CUDA toolkit      :", torch.version.cuda)
    print()

# --- tiny benchmark -----------------------------------------------------
def matmul_bench(device, size=6000, steps=3):
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    # warm-up
    _ = a @ b
    torch.cuda.synchronize() if device == "cuda" else None

    t0 = time.perf_counter()
    for _ in range(steps):
        _ = a @ b
    torch.cuda.synchronize() if device == "cuda" else None
    return (time.perf_counter() - t0) / steps

sizestr = "6000x6000"

print(f"Measuring average time for {sizestr} FP32 matmul …")

cpu_t = matmul_bench("cpu")
print(f"CPU  (torch.set_num_threads={torch.get_num_threads()}): {cpu_t:6.3f} s")

if torch.cuda.is_available():
    gpu_t = matmul_bench("cuda")
    speedup = cpu_t / gpu_t
    print(f"GPU  ({torch.cuda.get_device_name(0)}): {gpu_t:6.3f} s   -> {speedup:5.1f} × faster")
else:
    print("No CUDA device detected - run installs / drivers first!")
