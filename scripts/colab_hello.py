"""Diagnóstico mínimo del CLI de Colab: ¿asigna una VM (CPU)?"""
import platform
import subprocess

print("hello from colab:", platform.platform())
try:
    r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    print("gpu:", (r.stdout or r.stderr).strip() or "ninguna")
except FileNotFoundError:
    print("gpu: ninguna (nvidia-smi no existe)")
