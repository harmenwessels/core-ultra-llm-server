r"""Phase 0 / Gate 0: verify the Intel Arc iGPU is visible to OpenVINO.

Run with the project venv active:
    .\.venv\Scripts\python.exe scripts\check_gpu.py
"""

from __future__ import annotations

import subprocess
import sys

import openvino as ov


def print_driver_version() -> None:
    """Best-effort print of the Intel graphics DriverVersion via WMI (PowerShell)."""
    try:
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | "
                "Select-Object Name,DriverVersion,DriverDate | Format-List",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        print("--- Display adapter(s) (DriverVersion is the relevant one) ---")
        print(out.stdout.strip() or "(no output)")
        if out.stderr.strip():
            print(out.stderr.strip(), file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"(could not query driver via WMI: {exc})")


def main() -> int:
    print_driver_version()
    print()
    print(f"OpenVINO version: {ov.__version__}")
    core = ov.Core()
    devices = core.available_devices
    print(f"Available devices: {devices}")

    if "GPU" not in devices:
        print(
            "\nGATE 0 FAIL: no 'GPU' device exposed by OpenVINO.\n"
            "  -> Intel Arc driver may be outdated, or the OpenVINO GPU plugin "
            "did not load. Update the Intel Graphics driver and retry.",
            file=sys.stderr,
        )
        return 1

    try:
        full_name = core.get_property("GPU", "FULL_DEVICE_NAME")
    except Exception as exc:  # noqa: BLE001
        print(f"GATE 0 FAIL: could not read GPU FULL_DEVICE_NAME: {exc}", file=sys.stderr)
        return 1

    print(f"GPU FULL_DEVICE_NAME: {full_name}")

    name_lc = full_name.lower()
    is_arc = "arc" in name_lc or "xe" in name_lc or "meteor lake" in name_lc
    if not is_arc:
        print(
            "\nGATE 0 WARN: GPU device is exposed but the name does not clearly "
            "identify an Intel Arc iGPU. Inspect the FULL_DEVICE_NAME above before "
            "proceeding.",
            file=sys.stderr,
        )
        return 2

    print("\nGATE 0 PASS: Intel Arc iGPU is visible to OpenVINO.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
