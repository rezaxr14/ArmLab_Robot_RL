import ast
import json
import re
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "armforge_control_panel.ipynb"


def _cells():
    return json.loads(NB.read_text(encoding="utf-8"))["cells"]


def test_notebook_is_valid_json_with_outputs_cleared():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None


def test_every_code_cell_parses():
    for i, c in enumerate(_cells()):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        # drop IPython-only syntax (shell escapes and magics) before parsing
        py = "\n".join(re.sub(r"^\s*[!%].*$", "pass", line) for line in src.splitlines())
        try:
            ast.parse(py)
        except SyntaxError as exc:  # pragma: no cover
            raise AssertionError(f"cell {i} does not parse: {exc}\n{src}") from exc


def _module_symbols(module: str) -> set[str]:
    """Top-level names defined in a package module, found by parsing (no import needed)."""
    root = Path(__file__).resolve().parents[1] / "src"
    rel = Path(*module.split("."))
    path = root / rel.with_suffix(".py")
    if not path.exists():
        path = root / rel / "__init__.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
    return names


def test_notebook_imports_only_symbols_that_exist():
    src = "\n".join("".join(c["source"]) for c in _cells() if c["cell_type"] == "code")
    found = re.findall(r"from (armforge[\w.]*) import ([\w, ]+)", src)
    assert found, "notebook should import from armforge"
    for mod, names in found:
        defined = _module_symbols(mod)
        for n in [x.strip() for x in names.split(",") if x.strip()]:
            assert n in defined, f"{mod} has no symbol {n!r}"


def test_no_undefined_names_across_cells():
    """Cells run top to bottom in one namespace: every global a cell reads must exist by then."""
    import builtins

    defined = set(dir(builtins))
    for i, c in enumerate(_cells()):
        if c["cell_type"] != "code":
            continue
        src = "\n".join(re.sub(r"^\s*[!%].*$", "pass", line) for line in "".join(c["source"]).splitlines())
        tree = ast.parse(src)
        stores, loads = set(), []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Load):
                    loads.append(node.id)
                else:
                    stores.add(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                if not isinstance(node, ast.Lambda):
                    stores.add(node.name)
                a = node.args
                stores.update(x.arg for x in a.posonlyargs + a.args + a.kwonlyargs)
                stores.update(x.arg for x in (a.vararg, a.kwarg) if x)
            elif isinstance(node, ast.ClassDef):
                stores.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                stores.update((a.asname or a.name).split(".")[0] for a in node.names)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                stores.add(node.name)
        missing = sorted({n for n in loads if n not in defined and n not in stores})
        assert not missing, f"cell {i} reads names never defined before it: {missing}"
        defined |= stores


def test_setup_cell_makes_package_importable_in_a_running_kernel():
    """Regression: an editable install is invisible to an already-running Colab kernel, and a bare
    subprocess.run shows no output there. The setup cell must handle both."""
    setup = "".join(_cells()[1]["source"])
    assert "sys.path.insert(0, SRC_DIR)" in setup and "importlib.invalidate_caches()" in setup
    assert "import armforge" in setup
    assert "subprocess.Popen" in setup and "subprocess.run(cmd" not in setup  # output is streamed into the cell
