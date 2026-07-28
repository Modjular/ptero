"""Makes calls in cell/agent-authored source awaitable-if-needed, invisibly.

`maybe_sync()` in `_ptero_bridge.py` decides, per call, whether a wrapped shim method
runs synchronously (JS Promise Integration present) or returns a bare coroutine the
caller must `await` (JSPI absent). That's correct but not invisible: code copied from
real cellpose/stardist/instanseg docs -- or written by an LLM from its priors -- never
has `await` in it, and would silently receive an unawaited coroutine instead of a label
array on a browser without JSPI.

This fixes that by rewriting the *source*, not the runtime: every call in a cell is
wrapped in `await _ptero_auto_await(...)` before execution. `_ptero_auto_await` is a
no-op pass-through for ordinary values and transparently resolves a coroutine when one
comes back. Net effect: unmodified, upstream-style code (no `await`, exactly what a
human or an LLM would write from the real docs) produces correct results whether or not
JSPI is available -- `can_run_sync()` stays purely an implementation detail of
`maybe_sync`, never something cell code has to know about.

Scope limitation, by design: a user-defined *synchronous* helper that consumes a shim
result inline (`def analyze(img): masks, *_ = model.eval(img); return masks`) is not
fixed -- `await` is illegal inside a plain `def`/`lambda`/`class` body, and reaching
inside one would mean transitively colouring every call site `async`, which is exactly
the kind of open-ended, fragile transform this is trying to avoid. A helper that just
returns the shim call untouched (`def analyze(img): return model.eval(img)`) is fine:
the wrapping happens on the *value*, not the call site, so the coroutine passes through
and resolves once it reaches a wrapped call at the top level.

Known cost, accepted: `ast.unparse` doesn't preserve comments or exact line numbers, so a
*runtime* traceback's line number can land a few lines off from what's shown in the
editor. `SyntaxError`s are unaffected -- they're raised by `ast.parse` on the original
text, before any transform runs.
"""

import ast
import inspect

_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


class _AutoAwait(ast.NodeTransformer):
    """Wraps every call in `await _ptero_auto_await(...)`, except inside a new scope."""

    def visit_FunctionDef(self, node):
        return node  # don't recurse: `await` is illegal inside a plain def's body

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_Lambda = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef

    def visit_Call(self, node):
        self.generic_visit(node)  # transform nested calls in func/args/keywords first
        wrapper = ast.Call(
            func=ast.Name(id="_ptero_auto_await", ctx=ast.Load()),
            args=[node],
            keywords=[],
        )
        return ast.Await(value=wrapper)

    def visit_Await(self, node):
        # An explicit `await` (copied from an older non-JSPI example, or a coroutine
        # stashed in a variable and awaited later) has to be normalised through the same
        # single wrapper rather than left alone or blindly re-visited: leaving it alone
        # is wrong whenever `node.value` isn't a bare Call (`coro = f(); await coro`
        # would silently lose its await, since a bare Name never gets wrapped by
        # visit_Call), and letting the default traversal recurse into it is *also*
        # wrong, since visit_Call would then double-wrap the inner call, producing
        # `await (await _ptero_auto_await(...))` -- a TypeError on every browser, JSPI
        # or not. So: transform node.value's children (to catch nested calls in its own
        # arguments), then wrap node.value itself exactly once, without re-dispatching
        # it through visit_Call.
        node.value = self.generic_visit(node.value)
        node.value = ast.Call(
            func=ast.Name(id="_ptero_auto_await", ctx=ast.Load()),
            args=[node.value],
            keywords=[],
        )
        return node


def _ptero_auto_await_source(src):
    """Rewrite cell source so every call is awaited-if-needed, transparently."""
    tree = ast.parse(src, mode="exec")
    tree = _AutoAwait().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


async def _ptero_auto_await(value):
    """Pass through an ordinary value; resolve a coroutine if one comes back."""
    if inspect.isawaitable(value):
        return await value
    return value
