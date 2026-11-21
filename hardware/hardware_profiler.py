
import platform
import psutil
import shutil
import subprocess
import json
import sys
from pathlib import Path

from core.provenance import llama_cpp_version


def get_cpu_information():
    """Collect processor information."""

    return {
        "processor": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
    }


def get_memory_information():
    """Collect RAM information."""

    memory = psutil.virtual_memory()

    return {
        "total_ram_gb": round(memory.total / (1024 ** 3), 2),
        "available_ram_gb": round(memory.available / (1024 ** 3), 2),
        "used_ram_percent": memory.percent,
    }


def get_gpu_information():
    """Collect GPU information using Windows PowerShell."""

    try:
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name, AdapterRAM, DriverVersion | "
            "ConvertTo-Json"
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=15
        )

        if result.returncode != 0:
            return {"status": "GPU information unavailable"}

        gpu_data = json.loads(result.stdout)

        if isinstance(gpu_data, dict):
            gpu_data = [gpu_data]

        return {
            "devices": [
                {
                    "name": gpu.get("Name"),
                    "reported_adapter_memory_gb": (
                        round(gpu["AdapterRAM"] / (1024 ** 3), 2)
                        if gpu.get("AdapterRAM") else None
                    ),
                    "driver_version": gpu.get("DriverVersion")
                }
                for gpu in gpu_data
            ],
            "note": "Reported adapter memory is not necessarily dedicated VRAM."
        }

    except Exception as error:
        return {"status": f"GPU detection failed: {error}"}


def get_storage_information():
    """Collect storage information."""

    drive = Path.cwd().anchor or "C:\\"
    usage = shutil.disk_usage(drive)

    return {
        "drive": drive,
        "total_gb": round(usage.total / (1024 ** 3), 2),
        "free_gb": round(usage.free / (1024 ** 3), 2),
    }


def get_python_information():
    """Collect Python environment information."""

    return {
        "version": platform.python_version(),
        "executable": sys.executable,
    }


def get_backend_information():
    """Check whether inference backends are installed."""

    return {
        "llama_cpp_cli": shutil.which("llama-cli") is not None,
        "llama_cpp_server": shutil.which("llama-server") is not None,
        "ollama": shutil.which("ollama") is not None,
        "llama_cpp_version": llama_cpp_version(),
    }


def profile_hardware():
    """Generate the complete hardware profile."""

    return {
        "operating_system": platform.platform(),
        "cpu": get_cpu_information(),
        "memory": get_memory_information(),
        "gpu": get_gpu_information(),
        "storage": get_storage_information(),
        "python": get_python_information(),
        "inference_backends": get_backend_information(),
    }


if __name__ == "__main__":

    hardware_profile = profile_hardware()

    print("\n========== LOCAL LLM HARDWARE PROFILE ==========\n")

    print(json.dumps(
        hardware_profile,
        indent=4
    ))

    print("\n========== PROFILING COMPLETED ==========\n")
