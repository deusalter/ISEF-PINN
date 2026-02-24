"""
gmat_config.py -- GMAT path detection and validation
=====================================================

Locates the GMAT (General Mission Analysis Tool) installation on the system.

Search order:
  1. GMAT_PATH environment variable (from .env or shell)
  2. Default Windows install paths

Provides:
  - get_gmat_console_path()  -> Path to GmatConsole.exe
  - validate_gmat_install()  -> Run a trivial script to confirm GMAT works
"""

import os
import subprocess
from pathlib import Path

# Load .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

GMAT_DOWNLOAD_URL = "https://sourceforge.net/projects/gmat/files/GMAT/R2022a/"

# Default Windows installation paths (most common locations)
_DEFAULT_PATHS = [
    Path(r"C:\GMAT\R2022a\bin\GmatConsole.exe"),
    Path(r"C:\Program Files\GMAT\R2022a\bin\GmatConsole.exe"),
    Path(r"C:\Program Files (x86)\GMAT\R2022a\bin\GmatConsole.exe"),
    Path(os.path.expanduser("~")) / "GMAT" / "R2022a" / "bin" / "GmatConsole.exe",
    # R2020a fallbacks
    Path(r"C:\GMAT\R2020a\bin\GmatConsole.exe"),
    Path(r"C:\Program Files\GMAT\R2020a\bin\GmatConsole.exe"),
]


def get_gmat_console_path() -> Path:
    """Locate GmatConsole.exe on the system.

    Returns
    -------
    Path
        Absolute path to GmatConsole.exe.

    Raises
    ------
    FileNotFoundError
        If GMAT cannot be found.
    """
    # 1. Check GMAT_PATH env var
    env_path = os.environ.get("GMAT_PATH")
    if env_path:
        p = Path(env_path)
        # If user points to the GMAT root directory, append bin/GmatConsole.exe
        if p.is_dir():
            console = p / "bin" / "GmatConsole.exe"
            if console.exists():
                return console
        # If user points directly to the executable
        if p.is_file() and p.name.lower() == "gmatconsole.exe":
            return p
        # If user points to the bin directory
        if p.is_dir() and (p / "GmatConsole.exe").exists():
            return p / "GmatConsole.exe"

    # 2. Check default install paths
    for default in _DEFAULT_PATHS:
        if default.exists():
            return default

    raise FileNotFoundError(
        "GMAT installation not found.\n"
        "\n"
        "To fix this, either:\n"
        f"  1. Download GMAT from: {GMAT_DOWNLOAD_URL}\n"
        "  2. Set GMAT_PATH in your .env file, e.g.:\n"
        "       GMAT_PATH=C:\\GMAT\\R2022a\n"
        "  3. Or set the GMAT_PATH environment variable.\n"
    )


def get_gmat_root() -> Path:
    """Return the GMAT root directory (parent of bin/).

    Returns
    -------
    Path
        GMAT root directory containing bin/, data/, output/, etc.
    """
    console = get_gmat_console_path()
    # GmatConsole.exe is in <root>/bin/
    return console.parent.parent


def validate_gmat_install() -> bool:
    """Run a trivial GMAT script to verify the installation works.

    Creates a minimal script that just creates a spacecraft and propagates
    for 1 second. Returns True if GMAT exits successfully.

    Returns
    -------
    bool
        True if GMAT ran successfully, False otherwise.
    """
    try:
        console = get_gmat_console_path()
    except FileNotFoundError as e:
        print(f"GMAT validation failed: {e}")
        return False

    gmat_root = get_gmat_root()

    # Minimal GMAT validation script
    script_content = """%----------------------------------------
% Minimal GMAT validation script
% Just creates a spacecraft and reports success
%----------------------------------------

Create Spacecraft TestSat;
GMAT TestSat.DateFormat = UTCGregorian;
GMAT TestSat.Epoch = '01 Jan 2024 12:00:00.000';
GMAT TestSat.CoordinateSystem = EarthMJ2000Eq;
GMAT TestSat.X = 6778.137;
GMAT TestSat.Y = 0.0;
GMAT TestSat.Z = 0.0;
GMAT TestSat.VX = 0.0;
GMAT TestSat.VY = 7.6688;
GMAT TestSat.VZ = 0.0;

Create ForceModel fm;
GMAT fm.CentralBody = Earth;
GMAT fm.PrimaryBodies = {Earth};
GMAT fm.GravityField.Earth.Degree = 4;
GMAT fm.GravityField.Earth.Order = 4;

Create Propagator prop;
GMAT prop.FM = fm;
GMAT prop.Type = RungeKutta89;

Create Variable elapsed;
GMAT elapsed = 0.0;

BeginMissionSequence;
Propagate prop(TestSat) {TestSat.ElapsedSecs = 60.0};
"""
    # Write temp script
    script_path = gmat_root / "output" / "_validation_test.script"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script_content)

    try:
        result = subprocess.run(
            [str(console), "--run", "--exit", str(script_path)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(gmat_root),
        )
        success = result.returncode == 0
        if success:
            print("GMAT validation: OK")
        else:
            print(f"GMAT validation: FAILED (exit code {result.returncode})")
            if result.stderr:
                print(f"  stderr: {result.stderr[:500]}")
        return success
    except subprocess.TimeoutExpired:
        print("GMAT validation: FAILED (timeout after 60s)")
        return False
    except Exception as e:
        print(f"GMAT validation: FAILED ({e})")
        return False
    finally:
        # Clean up validation script
        if script_path.exists():
            script_path.unlink()


if __name__ == "__main__":
    print("=" * 60)
    print("  GMAT Configuration Check")
    print("=" * 60)

    try:
        path = get_gmat_console_path()
        print(f"\n  GmatConsole found: {path}")
        print(f"  GMAT root:         {get_gmat_root()}")
        print("\n  Running validation...")
        ok = validate_gmat_install()
        if ok:
            print("\n  GMAT is ready to use.")
        else:
            print("\n  GMAT was found but validation failed.")
    except FileNotFoundError as e:
        print(f"\n  {e}")
