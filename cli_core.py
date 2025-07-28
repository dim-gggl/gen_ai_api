#!/usr/bin/env python3
"""cli_core.py – shared, modern & reusable CLI framework
========================================================

Why this exists
---------------
You have *several* scripts (Anthropic, Gemini, OpenAI, …) which all roll their
own `argparse.ArgumentParser`.  That is brittle, verbose, and impossible to keep
in sync.  `cli_core` turns each script into a **pluggable sub-command** with a
single declarative decorator, while keeping 100 % stdlib (no Click/Typer
runtime dependency).  One import = all common defaults, validators and help
style.

Key features
~~~~~~~~~~~~
* **One-liner registration** → `@command("anthropic")`
* **Strict flags** (`allow_abbrev=False`) – no « --p » drama when you add
  `--port` later.
* **Dynamic discovery** – call ``discover_commands('text', 'gen_image')`` and
every module inside those packages that uses the decorator auto-registers.
* **Rich help** through a `CompactFormatter` (defaults + raw paragraphs).
* **Early validation** helpers (positive int, writable path …).
* **Graceful errors** (`exit_on_error=False`) so a library import won’t
  `sys.exit()` under your feet.
* **Typed** end-to-end (PEP 484) → editor autocompletion.

Usage sketch
~~~~~~~~~~~~
.. code-block:: python

    # main.py
    from cli_core import CLI, discover_commands

    discover_commands('text', 'gen_image')  # import side-effect: registers cmds

    cli = CLI(
        prog='toolkit',
        description='Swiss-army knife for LLMs',
        epilog='Run "toolkit <command> --help" for command-specific flags',
    )
    cli.run()

    # ``toolkit anthropic --help`` now delegates to text/anthropic_message.py

Each command module
-------------------

.. code-block:: python

    # text/anthropic_message.py  (trimmed)
    from cli_core import command, set_build_parser, positive_int


    @command('anthropic', help='Generate text via Anthropic Chat API')
    def main(args):
        ...  # your business logic here

    @set_build_parser('anthropic')
    def build(p):
        p.add_argument('--api-key', ...)
        p.add_argument('--max-tokens', type=positive_int, default=1024)

No more duplicate boilerplate, everything coherent.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import os
import pkgutil
import sys
import textwrap
from pathlib import Path
from typing import Callable, Dict, List, Optional

__all__ = [
    "CLI",
    "command",
    "set_build_parser",
    "positive_int",
    "writable_path",
]

# ---------------------------------------------------------------------------
# Validators (public helpers)
# ---------------------------------------------------------------------------

def positive_int(value: str) -> int:
    """Return *value* as ``int`` if > 0, else raise ``ArgumentTypeError``."""

    try:
        ivalue = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from exc

    if ivalue <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return ivalue


def writable_path(value: str) -> Path:
    """Ensure *value* is a user-writable file path (existing or not)."""

    p = Path(value).expanduser()
    if p.exists():
        if not p.is_file():
            raise argparse.ArgumentTypeError(f"{value!r} exists and is not a file")
        if not os.access(p, os.W_OK):
            raise argparse.ArgumentTypeError(f"{value!r} is not writable")
    else:
        parent = p.parent
        if not parent.exists() or not os.access(parent, os.W_OK):
            raise argparse.ArgumentTypeError(
                f"Parent directory {parent!r} is not writable or does not exist"
            )
    return p


# ---------------------------------------------------------------------------
# Internal plumbing – registry & decorators
# ---------------------------------------------------------------------------

CommandHandler = Callable[[argparse.Namespace], None]
BuildParserFn = Callable[[argparse.ArgumentParser], None]


@dataclasses.dataclass
class _CommandSpec:
    name: str
    handler: CommandHandler
    build_parser: Optional[BuildParserFn]
    help: str = ""


_COMMANDS: Dict[str, _CommandSpec] = {}


def command(name: str, *, help: str = "") -> Callable[[CommandHandler], CommandHandler]:
    """Decorator registering *name* as a sub-command handler."""

    def decorator(func: CommandHandler) -> CommandHandler:
        if name in _COMMANDS:
            raise RuntimeError(f"Command {name!r} already registered")
        _COMMANDS[name] = _CommandSpec(name, func, build_parser=None, help=help)
        return func

    return decorator


def set_build_parser(name: str) -> Callable[[BuildParserFn], BuildParserFn]:
    """Attach a *build_parser* callback to an existing command."""

    def decorator(fn: BuildParserFn) -> BuildParserFn:
        try:
            spec = _COMMANDS[name]
        except KeyError as exc:
            raise RuntimeError(
                f"Command {name!r} must be registered via @command before "
                "@set_build_parser"
            ) from exc
        if spec.build_parser is not None:
            raise RuntimeError(f"Command {name!r} already has a build_parser")
        spec.build_parser = fn
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Help formatter (defaults + raw text)
# ---------------------------------------------------------------------------

class CompactFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    pass


# ---------------------------------------------------------------------------
# CLI dispatcher (public API)
# ---------------------------------------------------------------------------

class CLI:
    """Root command-line interface dispatcher.

    Parameters
    ----------
    prog:
        Display name used in the help output.  Defaults to ``Path(sys.argv[0]).name``.
    description:
        Short text shown below the usage synopsis.
    epilog:
        Extra text appended at the bottom of ``--help``.
    """

    def __init__(
        self,
        *,
        prog: Optional[str] = None,
        description: str = "",
        epilog: Optional[str] = None,
        version: str | None = None,
    ):
        self._version = version
        self.parser = argparse.ArgumentParser(
            prog=prog or Path(sys.argv[0]).name,
            description=textwrap.dedent(description),
            epilog=textwrap.dedent(epilog) if epilog else None,
            formatter_class=CompactFormatter,
            allow_abbrev=False,
            fromfile_prefix_chars="@",
            argument_default=argparse.SUPPRESS,
            conflict_handler="resolve",
            exit_on_error=False,
        )
        if version is not None:
            self.parser.add_argument(
                "-V", "--version", action="version", version=f"%(prog)s {version}"
            )

        self.subparsers = self.parser.add_subparsers(
            dest="command",
            metavar="{cmd}",  # will be replaced later when commands exist
            required=True,
        )
        self._populate_subparsers()

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------

    def run(self, argv: List[str] | None = None) -> None:  # pragma: no cover
        """Parse *argv* (defaults to ``sys.argv[1:]``) and dispatch."""

        try:
            ns = self.parser.parse_args(argv)
        except argparse.ArgumentError as exc:
            # Convert to the classical error output & exit(2)
            self.parser.error(str(exc))

        handler: CommandHandler = getattr(ns, "_handler")
        handler(ns)

    # ---------------------------------------------------------------------
    # Private – build sub-parsers
    # ---------------------------------------------------------------------

    def _populate_subparsers(self) -> None:
        if not _COMMANDS:
            self.subparsers.metavar = "{no-commands-found}"
            return

        self.subparsers.metavar = "{" + ",".join(sorted(_COMMANDS)) + "}"

        for name, spec in sorted(_COMMANDS.items()):
            sub = self.subparsers.add_parser(
                name,
                help=spec.help or spec.handler.__doc__,
                formatter_class=CompactFormatter,
            )
            if spec.build_parser is not None:
                spec.build_parser(sub)
            sub.set_defaults(_handler=spec.handler)


# ---------------------------------------------------------------------------
# Discovery helper – import packages to auto-register commands
# ---------------------------------------------------------------------------

def discover_commands(*package_names: str) -> None:
    """Import every module inside *package_names* to trigger decorator side-effects."""

    for pkg_name in package_names:
        try:
            pkg = importlib.import_module(pkg_name)
        except ModuleNotFoundError:
            continue

        # If it’s a namespace or normal package, recurse into sub-modules
        if hasattr(pkg, "__path__"):
            for mod in pkgutil.walk_packages(pkg.__path__, prefix=f"{pkg_name}."):
                importlib.import_module(mod.name)
