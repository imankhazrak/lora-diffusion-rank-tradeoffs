#!/usr/bin/env python3
import platform
import sys


def safe_import(name):
    try:
        module = __import__(name)
        return module, None
    except Exception as exc:  # pragma: no cover - operational script
        return None, str(exc)


def main():
    print(f"Python version: {sys.version.splitlines()[0]}")
    print(f"Platform: {platform.platform()}")

    torch, torch_err = safe_import("torch")
    if torch is None:
        print(f"torch version: UNAVAILABLE ({torch_err})")
        print("CUDA available: False")
    else:
        print(f"torch version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        if cuda_available:
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU name: N/A")

    for pkg in ("diffusers", "accelerate", "datasets"):
        module, err = safe_import(pkg)
        if module is None:
            print(f"{pkg} version: UNAVAILABLE ({err})")
        else:
            print(f"{pkg} version: {module.__version__}")


if __name__ == "__main__":
    main()
