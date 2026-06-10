import ast
import json
import logging
import operator
from typing import Any

from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)

_ALLOWED_BINOPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARYOPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval_node(node: ast.AST) -> float:
    """Evaluate a restricted arithmetic AST node without calling eval()."""
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Num):  # pragma: no cover - py<3.8 compat
        return float(node.n)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return float(_ALLOWED_UNARYOPS[type(node.op)](_safe_eval_node(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        return float(_ALLOWED_BINOPS[type(node.op)](left, right))
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def _evaluate_expression(expression: str) -> float:
    """Parse and evaluate a math expression using a restricted AST walker."""
    try:
        import numexpr as ne

        result = ne.evaluate(expression.strip())
        return float(result.item() if hasattr(result, "item") else result)
    except Exception:
        tree = ast.parse(expression.strip(), mode="eval")
        return _safe_eval_node(tree)


def calculator_tool(expression: str) -> str:
    """
    Evaluate a safe arithmetic math expression and return the numeric result.

    Use this tool for calculations involving numbers and operators (+, -, *, /, //, %, **).
    Only arithmetic is supported — no variables, functions, or code execution.
    """
    try:
        value = _evaluate_expression(expression)
        return json.dumps({"expression": expression, "result": value}, ensure_ascii=False)
    except ZeroDivisionError:
        return json.dumps(
            {"expression": expression, "error": "Division by zero."},
            ensure_ascii=False,
        )
    except Exception as exc:
        logger.warning("Calculator failed for %r: %s", expression, exc)
        return json.dumps(
            {"expression": expression, "error": f"Invalid expression: {exc}"},
            ensure_ascii=False,
        )


def create_calculator_tool() -> StructuredTool:
    """Factory for the safe calculator LangChain tool."""
    return StructuredTool.from_function(
        func=calculator_tool,
        name="calculator_tool",
        description=(
            "Evaluate arithmetic math expressions safely. "
            "Use for numeric calculations with +, -, *, /, //, %, and **."
        ),
    )
