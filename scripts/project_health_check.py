import importlib
import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_PYTHON_VERSION = "3.10.11"
EXPECTED_VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

FOUNDATIONAL_PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("jupyter", "jupyter"),
    ("openpyxl", "openpyxl"),
    ("tqdm", "tqdm"),
    ("yaml", "PyYAML"),
    ("networkx", "networkx"),
    ("xgboost", "xgboost"),
]

REQUIRED_DIRECTORIES = [
    "data/raw",
    "data/interim",
    "data/processed",
    "docs",
    "ml/preprocessing",
    "ml/graph",
    "ml/baselines",
    "ml/models",
    "ml/uncertainty",
    "ml/risk",
    "ml/path_analysis",
    "ml/evaluation",
    "backend/api",
    "backend/services",
    "backend/models",
    "backend/config",
    "frontend",
    "experiments/configs",
    "experiments/results",
    "experiments/figures",
    "notebooks",
    "tests",
    "scripts",
    "checkpoints",
]

REQUIRED_FILES = [
    "README.md",
    ".gitignore",
    "docs/architecture.md",
    "docs/threat_model.md",
    "docs/research_question.md",
    "docs/data_dictionary.md",
    "docs/research_gap.md",
    "docs/experiment_plan.md",
    "requirements-base.txt",
    "scripts/package_check.py",
]

FORBIDDEN_MODULES = [
    ("torch", "PyTorch"),
    ("torch_geometric", "PyTorch Geometric"),
]

LINE = "=" * 64


def check_python_version(results):
    actual = ".".join(str(part) for part in sys.version_info[:3])
    ok = actual == EXPECTED_PYTHON_VERSION
    results.append(("Python version", ok,
                    f"expected={EXPECTED_PYTHON_VERSION} found={actual}"))


def check_python_executable(results):
    exe = Path(sys.executable).resolve()
    expected = EXPECTED_VENV_PYTHON.resolve()
    same_file = exe.name.lower() == "python.exe" and _paths_equal(exe, expected)
    inside_venv = ".venv" in exe.parts[:-1]
    ok = same_file or inside_venv
    results.append(("Python executable", ok,
                    f"found={exe}"))


def _paths_equal(a, b):
    return str(a).lower() == str(b).lower()


def check_virtual_environment(results):
    in_venv = hasattr(sys, "prefix") and sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    prefix_is_project_venv = ".venv" in Path(sys.prefix).parts
    ok = in_venv and prefix_is_project_venv
    results.append(("Virtual environment", ok,
                    f"prefix={sys.prefix}"))


def check_foundational_packages(results):
    failures = []
    for module_name, dist_name in FOUNDATIONAL_PACKAGES:
        try:
            importlib.import_module(module_name)
            importlib.metadata.version(dist_name)
        except Exception as exc:
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")
    ok = not failures
    detail = f"imported {len(FOUNDATIONAL_PACKAGES) - len(failures)}/{len(FOUNDATIONAL_PACKAGES)}"
    if failures:
        detail += "; failed: " + ", ".join(failures)
    results.append(("Foundational packages", ok, detail))


def check_directories(results):
    missing = []
    for rel in REQUIRED_DIRECTORIES:
        if not (PROJECT_ROOT / rel).is_dir():
            missing.append(rel)
    ok = not missing
    detail = f"{len(REQUIRED_DIRECTORIES) - len(missing)}/{len(REQUIRED_DIRECTORIES)} present"
    if missing:
        detail += "; MISSING: " + ", ".join(missing)
    results.append(("Project directories", ok, detail))


def check_files(results):
    missing = []
    for rel in REQUIRED_FILES:
        if not (PROJECT_ROOT / rel).is_file():
            missing.append(rel)
    ok = not missing
    detail = f"{len(REQUIRED_FILES) - len(missing)}/{len(REQUIRED_FILES)} present"
    if missing:
        detail += "; MISSING: " + ", ".join(missing)
    results.append(("Documentation & required files", ok, detail))
    req_ok = (PROJECT_ROOT / "requirements-base.txt").is_file()
    results.append(("Requirements file", req_ok, "requirements-base.txt"))
    doc_files = [f for f in REQUIRED_FILES if f.startswith("docs/") or f == "README.md"]
    docs_missing = [f for f in doc_files if not (PROJECT_ROOT / f).is_file()]
    results.append(("Documentation", not docs_missing,
                    "all present" if not docs_missing else "MISSING: " + ", ".join(docs_missing)))


def check_forbidden_modules(results):
    for module_name, label in FORBIDDEN_MODULES:
        spec = importlib.util.find_spec(module_name)
        absent = spec is None
        results.append((f"{label} not installed", absent,
                        "absent (expected)" if absent else f"FOUND at {spec.origin}"))


def main():
    results = []

    print(LINE)
    print("PROJECT HEALTH CHECK")
    print(LINE)
    print(f"Project root      : {PROJECT_ROOT}")
    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {'.'.join(str(p) for p in sys.version_info[:3])}")
    print(LINE)

    check_python_version(results)
    check_python_executable(results)
    check_virtual_environment(results)
    check_foundational_packages(results)
    check_directories(results)
    check_files(results)
    check_forbidden_modules(results)

    print()
    for name, ok, detail in results:
        status = "[PASS]" if ok else "[FAIL]"
        print(f"{status} {name}")
        print(f"       {detail}")

    print(LINE)
    total = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"Checks passed: {passed}/{total}")
    if passed == total:
        print("RESULT: PASS")
        return 0
    print("RESULT: FAIL")
    return 1


if __name__ == "__main__":
    os.system("")
    sys.exit(main())
