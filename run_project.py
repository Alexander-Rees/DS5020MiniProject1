"""Rebuild the matrix, models, reports, figures, tests and completion notebook in one go.

    python run_project.py                 # matrix from the raw file, analysis (cached fits reused), tests, notebook
    python run_project.py --skip-matrix   # same, reusing data/processed instead of re-reading the Excel file
    python run_project.py --force         # also refit every model in fresh processes

Steps: build_matrix.py -> model_analysis.py (reports/ and figures/) -> pytest -> execute
notebooks/model_analysis.ipynb in place (outputs saved in the notebook). Every step, including the
notebook kernel, runs with this interpreter (sys.executable), so start it from the project .venv.

Not included: memo.pdf and the slides. They are built separately from reports/results.json
(see the README, "Memo and slides").
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "notebooks" / "model_analysis.ipynb"


def run(*args):
    print(f"\n$ python {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def current_interpreter_kernel_manager():
    """A kernel manager whose python3 kernel is started with sys.executable.

    The installed python3 kernelspec says just "python", which would pick up whatever python is first
    on PATH; this pins the kernel to the interpreter running this script.
    """
    from ipykernel.kernelspec import get_kernel_dict
    from jupyter_client.kernelspec import KernelSpec
    from jupyter_client.manager import KernelManager

    class CurrentInterpreterKernelManager(KernelManager):
        @property
        def kernel_spec(self):
            return KernelSpec(resource_dir="", **get_kernel_dict())    # argv[0] is sys.executable

    return CurrentInterpreterKernelManager


def execute_notebook(path=NOTEBOOK, timeout=1800):
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=timeout, kernel_name="python3",
                   kernel_manager_class=current_interpreter_kernel_manager(),
                   resources={"metadata": {"path": str(path.parent)}}).execute()
    nbformat.write(nb, path)
    print(f"executed and saved {path.relative_to(ROOT)} with {sys.executable}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="refit every model (passed to model_analysis.py)")
    parser.add_argument("--skip-matrix", action="store_true", help="reuse data/processed instead of re-reading Excel")
    args = parser.parse_args(argv)
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ds5020-mpl")
    if not args.skip_matrix:
        run("build_matrix.py")
    run("model_analysis.py", *(["--force"] if args.force else []))
    run("-m", "pytest", "-q")
    print("\n$ execute notebooks/model_analysis.ipynb", flush=True)
    execute_notebook()


if __name__ == "__main__":
    main()
