#!/usr/bin/env python3
# pip install libcst

import sys
from pathlib import Path

import libcst as cst


# Only annotate normal functions and __init__
def _should_consider(name: str) -> bool:
    if name == "__init__":
        return True
    if name.startswith("__") and name.endswith("__"):
        return False
    return True


class _ReturnValueFinder(cst.CSTVisitor):
    def __init__(self) -> None:
        self.has_value_return = False

    def visit_Return(self, node: cst.Return) -> None:
        if node.value is not None:
            self.has_value_return = True

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        return False  # do not look inside nested defs

    def visit_Lambda(self, node: cst.Lambda) -> bool:
        return False


class AddNoneReturn(cst.CSTTransformer):
    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ):
        # already annotated
        if updated_node.returns is not None:
            return updated_node

        name = updated_node.name.value
        if not _should_consider(name):
            return updated_node

        finder = _ReturnValueFinder()
        original_node.visit(finder)
        if not finder.has_value_return:
            return updated_node.with_changes(
                returns=cst.Annotation(annotation=cst.Name("None"))
            )
        return updated_node


def _rewrite_file(path: Path) -> None:
    try:
        src = path.read_text(encoding="utf-8")
        mod = cst.parse_module(src)
        new = mod.visit(AddNoneReturn())
        if new.code != src:
            path.write_text(new.code, encoding="utf-8")
    except Exception:
        pass  # skip unreadable or syntactically invalid files


def main(args: list[str]) -> None:
    if len(args) < 1:
        print("Usage: python add_none_returns.py <path> [<path> ...]")
        sys.exit(1)
    for root in args:
        p = Path(root)
        files = []
        if p.is_dir():
            files = [
                f
                for f in p.rglob("*.py")
                if ".venv" not in str(f) and "venv" not in str(f)
            ]
        elif p.suffix == ".py":
            files = [p]
        for f in files:
            _rewrite_file(f)


if __name__ == "__main__":
    main(sys.argv[1:])
