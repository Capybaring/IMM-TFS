#!/usr/bin/env python3
"""
Install Expanded-MIMIC v2 files into an IMM-TFS checkout.

Run from the repository root:
    python apply_expanded_mimic_v2.py

Actions
-------
1. backs up and replaces:
     lib/parse_datasets_mimic_expanded.py
     compute_text_embeddings.py
     scripts/_common.sh
2. patches lib/evaluation.py so GPINet native text receives tau_raw while
   generic FusionModel continues to receive normalized tau.
3. ensures main.py imports the expanded-MIMIC adapter.
4. performs Python syntax checks.

No dataset files are created, moved, deleted, or symlinked.
"""

from pathlib import Path
import py_compile
import shutil
import sys

ROOT = Path.cwd()
BUNDLE = Path(__file__).resolve().parent

required_repo = [
    ROOT / "main.py",
    ROOT / "lib",
    ROOT / "scripts",
    ROOT / "lib" / "evaluation.py",
]
if not all(p.exists() for p in required_repo):
    raise SystemExit(
        "Run from the IMM-TFS repository root (main.py/lib/scripts required)."
    )

replacements = {
    ROOT / "lib" / "parse_datasets_mimic_expanded.py":
        BUNDLE / "lib" / "parse_datasets_mimic_expanded.py",
    ROOT / "compute_text_embeddings.py":
        BUNDLE / "compute_text_embeddings.py",
    ROOT / "scripts" / "_common.sh":
        BUNDLE / "scripts" / "_common.sh",
}


def backup_once(path: Path):
    backup = path.with_name(path.name + ".before_expanded_mimic_v2")
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)
        print(f"Backup: {backup}")


for target, source in replacements.items():
    if not source.exists():
        raise SystemExit(f"Bundle file missing: {source}")
    backup_once(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(f"Installed: {target}")

# Patch evaluation.py: only GPINet native path needs raw-hour tau.
evaluation = ROOT / "lib" / "evaluation.py"
backup_once(evaluation)
text = evaluation.read_text(encoding="utf-8")

old = 'tau=batch_dict["tau"],'
new = 'tau=batch_dict.get("tau_raw", batch_dict["tau"]),'

occurrences = text.count(old)
if occurrences == 0:
    if new in text:
        print("evaluation.py tau_raw routing already patched.")
    else:
        raise SystemExit(
            "Could not find expected GPINet tau call(s) in lib/evaluation.py. "
            "No evaluation changes were made."
        )
else:
    # Current repository has two GPINet-native call sites:
    # compute_all_losses() and evaluation().
    text = text.replace(old, new)
    evaluation.write_text(text, encoding="utf-8")
    print(
        f"Patched lib/evaluation.py: {occurrences} native GPINet tau call(s) "
        "now prefer tau_raw."
    )

# Ensure main.py dispatches through the expanded adapter.
main = ROOT / "main.py"
backup_once(main)
main_text = main.read_text(encoding="utf-8")

old_import = (
    "from lib.parse_datasets import parse_datasets, get_input_and_pred_len"
)
new_import = (
    "from lib.parse_datasets_mimic_expanded import "
    "parse_datasets, get_input_and_pred_len"
)

if new_import in main_text:
    print("main.py expanded-MIMIC import already present.")
elif old_import in main_text:
    main_text = main_text.replace(old_import, new_import, 1)
    main.write_text(main_text, encoding="utf-8")
    print("Patched main.py expanded-MIMIC import.")
else:
    raise SystemExit(
        "Could not find the expected parse_datasets import in main.py."
    )

# Syntax checks.
for p in [
    ROOT / "lib" / "parse_datasets_mimic_expanded.py",
    ROOT / "compute_text_embeddings.py",
    ROOT / "lib" / "evaluation.py",
    ROOT / "main.py",
]:
    py_compile.compile(str(p), doraise=True)
    print(f"Syntax OK: {p}")

print("\nExpanded-MIMIC v2 installation complete.")
print("No data paths or symlinks were changed.")
print("\nRecommended checks:")
print("  ./scripts/run_gpinet.sh --smoke")
print("Then, only after numeric smoke succeeds:")
print("  ./scripts/run_gpinet.sh --smoke --text")
