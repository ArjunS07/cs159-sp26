"""EGL vendor registration: the difference between GPU and CPU offscreen rendering.

Colab ships only 50_mesa.json, so MUJOCO_GL=egl silently resolves to Mesa software rendering.
Measured on an L4 that made the simulator ~90% of a rollout (~142 ms/step vs ~6 ms of policy
inference), so these guards are about wall-clock correctness, not cosmetics.
"""
import json

from pnp.env_setup import (
    _ensure_nvidia_egl_vendor,
    _find_nvidia_egl_library,
    egl_vendors_label,
)


def _mesa_only(vendor_dir):
    vendor_dir.mkdir(parents=True, exist_ok=True)
    (vendor_dir / "50_mesa.json").write_text("{}")
    return vendor_dir


def test_registers_nvidia_icd_when_only_mesa_is_present(tmp_path):
    vendor_dir = _mesa_only(tmp_path / "egl_vendor.d")
    driver_dir = tmp_path / "lib"
    driver_dir.mkdir()
    (driver_dir / "libEGL_nvidia.so.0").write_bytes(b"")

    assert _ensure_nvidia_egl_vendor(str(vendor_dir), [str(driver_dir)]) is True
    icd = json.loads((vendor_dir / "10_nvidia.json").read_text())
    assert icd["ICD"]["library_path"] == "libEGL_nvidia.so.0"
    # 10_ must sort before 50_mesa or the dispatcher still prefers software rendering.
    assert sorted(p.name for p in vendor_dir.iterdir())[0] == "10_nvidia.json"


def test_never_registers_an_icd_whose_library_is_missing(tmp_path):
    """Pointing EGL at an absent vendor is worse than leaving Mesa in place."""
    vendor_dir = _mesa_only(tmp_path / "egl_vendor.d")
    assert _ensure_nvidia_egl_vendor(str(vendor_dir), [str(tmp_path / "empty")]) is False
    assert not (vendor_dir / "10_nvidia.json").exists()


def test_is_idempotent(tmp_path):
    vendor_dir = tmp_path / "egl_vendor.d"
    vendor_dir.mkdir()
    (vendor_dir / "10_nvidia.json").write_text('{"existing": true}')
    driver_dir = tmp_path / "lib"
    driver_dir.mkdir()
    (driver_dir / "libEGL_nvidia.so.0").write_bytes(b"")

    assert _ensure_nvidia_egl_vendor(str(vendor_dir), [str(driver_dir)]) is False
    # An operator-supplied ICD is left exactly as found.
    assert json.loads((vendor_dir / "10_nvidia.json").read_text()) == {"existing": True}


def test_missing_vendor_directory_is_created(tmp_path):
    vendor_dir = tmp_path / "nested" / "egl_vendor.d"
    driver_dir = tmp_path / "lib"
    driver_dir.mkdir()
    (driver_dir / "libEGL_nvidia.so.0").write_bytes(b"")
    assert _ensure_nvidia_egl_vendor(str(vendor_dir), [str(driver_dir)]) is True
    assert (vendor_dir / "10_nvidia.json").exists()


def test_library_search_prefers_the_first_matching_directory(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    for directory in (first, second):
        directory.mkdir()
        (directory / "libEGL_nvidia.so.0").write_bytes(b"")
    assert _find_nvidia_egl_library([str(first), str(second)]).startswith(str(first))
    assert _find_nvidia_egl_library([str(tmp_path / "missing")]) is None


def test_vendor_label_flags_cpu_only_configurations(tmp_path):
    vendor_dir = _mesa_only(tmp_path / "egl_vendor.d")
    assert egl_vendors_label(str(vendor_dir)) == "50_mesa(CPU!)"

    (vendor_dir / "10_nvidia.json").write_text("{}")
    assert egl_vendors_label(str(vendor_dir)) == "10_nvidia+50_mesa"

    assert egl_vendors_label(str(tmp_path / "absent")) == "none"
