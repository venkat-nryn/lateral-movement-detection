import importlib
import importlib.metadata
import sys

PACKAGES = [
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

LINE = "-" * 64


def main():
    print(LINE)
    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {sys.version.split()[0]}")
    print(LINE)

    failures = []
    for module_name, dist_name in PACKAGES:
        try:
            module = importlib.import_module(module_name)
            version = importlib.metadata.version(dist_name)
            print(f"PASS  {module_name:<12} import ok   dist={dist_name:<14} version={version}")
        except Exception as exc:
            failures.append((module_name, dist_name, exc))
            print(f"FAIL  {module_name:<12} import error dist={dist_name:<14} "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)

    print(LINE)
    total = len(PACKAGES)
    passed = total - len(failures)
    print(f"Imported successfully: {passed}/{total}")

    if failures:
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
