#!/usr/bin/env python3
"""Install the fixed/nested expanded-MIMIC v3 protocol into a local IMM-TSF checkout.

Run from the repository root:
    python /path/to/imm_tfs_mimic_fixed_protocol_v3/apply_mimic_fixed_protocol_v3.py

This installer does NOT read GitHub and does NOT modify dataset files.
"""

from pathlib import Path
import py_compile
import shutil

ROOT = Path.cwd()
BUNDLE = Path(__file__).resolve().parent

required = [ROOT / "main.py", ROOT / "lib", ROOT / "scripts", ROOT / "lib" / "evaluation.py"]
if not all(p.exists() for p in required):
    raise SystemExit("Run this installer from the local IMM-TSF repository root.")

replacements = {
    ROOT / "lib" / "parse_datasets_mimic_expanded.py": BUNDLE / "lib" / "parse_datasets_mimic_expanded.py",
    ROOT / "compute_text_embeddings.py": BUNDLE / "compute_text_embeddings.py",
    ROOT / "scripts" / "prepare_mimic_fixed_protocol.py": BUNDLE / "scripts" / "prepare_mimic_fixed_protocol.py",
    ROOT / "scripts" / "run_gpinet_fixed.sh": BUNDLE / "scripts" / "run_gpinet_fixed.sh",
}


def backup_once(path: Path):
    if not path.exists():
        return
    backup = path.with_name(path.name + ".before_mimic_fixed_v3")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"Backup: {backup}")


for target, source in replacements.items():
    if not source.exists():
        raise SystemExit(f"Bundle file missing: {source}")
    backup_once(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if target.suffix == ".sh" or target.name.startswith("prepare_"):
        target.chmod(target.stat().st_mode | 0o111)
    print(f"Installed: {target}")

# Keep local main.py routing through the expanded adapter.
main = ROOT / "main.py"
backup_once(main)
main_text = main.read_text(encoding="utf-8")
old_import = "from lib.parse_datasets import parse_datasets, get_input_and_pred_len"
new_import = "from lib.parse_datasets_mimic_expanded import parse_datasets, get_input_and_pred_len"
if new_import in main_text:
    print("main.py expanded-MIMIC import already present.")
elif old_import in main_text:
    main.write_text(main_text.replace(old_import, new_import, 1), encoding="utf-8")
    print("Patched main.py expanded-MIMIC import.")
else:
    raise SystemExit("Could not find expected parse_datasets import in local main.py")

# Preserve the v2 raw-hour tau fix for GPINet native text fusion if needed.
evaluation = ROOT / "lib" / "evaluation.py"
backup_once(evaluation)
ev_text = evaluation.read_text(encoding="utf-8")
old_tau = 'tau=batch_dict["tau"],'
new_tau = 'tau=batch_dict.get("tau_raw", batch_dict["tau"]),'
if old_tau in ev_text:
    n = ev_text.count(old_tau)
    evaluation.write_text(ev_text.replace(old_tau, new_tau), encoding="utf-8")
    print(f"Patched evaluation.py tau_raw routing at {n} call(s).")
elif new_tau in ev_text:
    print("evaluation.py tau_raw routing already present.")
else:
    print("[WARN] evaluation.py GPINet tau call pattern not found; left unchanged.")

for p in [
    ROOT / "lib" / "parse_datasets_mimic_expanded.py",
    ROOT / "compute_text_embeddings.py",
    ROOT / "scripts" / "prepare_mimic_fixed_protocol.py",
    ROOT / "main.py",
    ROOT / "lib" / "evaluation.py",
]:
    py_compile.compile(str(p), doraise=True)
    print(f"Syntax OK: {p}")

print("\nFixed/nested expanded-MIMIC v3 (per-N normalization) installed.")
print("Existing data/MIMIC contents were not modified.")
print("Existing scripts/run_gpinet.sh is intentionally untouched.")
print("\nNext:")
print("  python scripts/prepare_mimic_fixed_protocol.py")
print("  ./scripts/run_gpinet_fixed.sh -n 200")
print("  ./scripts/run_gpinet_fixed.sh -n 200 --text")
