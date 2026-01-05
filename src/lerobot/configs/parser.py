# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import inspect
import pkgutil
import sys
from argparse import ArgumentError
from collections.abc import Callable, Iterable, Sequence
from functools import wraps
from pathlib import Path
from pkgutil import ModuleInfo
from types import ModuleType
from typing import Any, TypeVar, cast

import draccus

from lerobot.utils.utils import has_method

F = TypeVar("F", bound=Callable[..., object])

PATH_KEY = "path"
PLUGIN_DISCOVERY_SUFFIX = "discover_packages_path"


def get_cli_overrides(field_name: str, args: Sequence[str] | None = None) -> list[str] | None:
    """Parses arguments from cli at a given nested attribute level.

    For example, supposing the main script was called with:
    python myscript.py --arg1=1 --arg2.subarg1=abc --arg2.subarg2=some/path

    If called during execution of myscript.py, get_cli_overrides("arg2") will return:
    ["--subarg1=abc" "--subarg2=some/path"]
    """
    if args is None:
        args = sys.argv[1:]
    attr_level_args = []
    detect_string = f"--{field_name}."
    exclude_strings = (f"--{field_name}.{draccus.CHOICE_TYPE_KEY}=", f"--{field_name}.{PATH_KEY}=")
    for arg in args:
        if arg.startswith(detect_string) and not arg.startswith(exclude_strings):
            denested_arg = f"--{arg.removeprefix(detect_string)}"
            attr_level_args.append(denested_arg)

    return attr_level_args


def parse_arg(arg_name: str, args: Sequence[str] | None = None) -> str | None:
    if args is None:
        args = sys.argv[1:]
    prefix = f"--{arg_name}="
    for arg in args:
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def parse_plugin_args(plugin_arg_suffix: str, args: Sequence[str]) -> dict[str, str]:
    """Parse plugin-related arguments from command-line arguments.

    This function extracts arguments from command-line arguments that match a specified suffix pattern.
    It processes arguments in the format '--key=value' and returns them as a dictionary.

    Args:
        plugin_arg_suffix (str): The suffix to identify plugin-related arguments.
        cli_args (Sequence[str]): A sequence of command-line arguments to parse.

    Returns:
        dict: A dictionary containing the parsed plugin arguments where:
            - Keys are the argument names (with '--' prefix removed if present)
            - Values are the corresponding argument values

    Example:
        >>> args = ["--env.discover_packages_path=my_package", "--other_arg=value"]
        >>> parse_plugin_args("discover_packages_path", args)
        {'env.discover_packages_path': 'my_package'}
    """
    plugin_args = {}
    for arg in args:
        if "=" in arg and plugin_arg_suffix in arg:
            key, value = arg.split("=", 1)
            # Remove leading '--' if present
            if key.startswith("--"):
                key = key[2:]
            plugin_args[key] = value
    return plugin_args


class PluginLoadError(Exception):
    """Raised when a plugin fails to load."""


def load_plugin(plugin_path: str) -> None:
    """Load and initialize a plugin from a given Python package path.

    This function attempts to load a plugin by importing its package and any submodules.
    Plugin registration is expected to happen during package initialization, i.e. when
    the package is imported the gym environment should be registered and the config classes
    registered with their parents using the `register_subclass` decorator.

    Args:
        plugin_path (str): The Python package path to the plugin (e.g. "mypackage.plugins.myplugin")

    Raises:
        PluginLoadError: If the plugin cannot be loaded due to import errors or if the package path is invalid.

    Examples:
        >>> load_plugin("external_plugin.core")  # Loads plugin from external package

    Notes:
        - The plugin package should handle its own registration during import
        - All submodules in the plugin package will be imported
        - Implementation follows the plugin discovery pattern from Python packaging guidelines

    See Also:
        https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/
    """
    try:
        package_module = importlib.import_module(plugin_path, __package__)
    except (ImportError, ModuleNotFoundError) as e:
        raise PluginLoadError(
            f"Failed to load plugin '{plugin_path}'. Verify the path and installation: {str(e)}"
        ) from e

    def iter_namespace(ns_pkg: ModuleType) -> Iterable[ModuleInfo]:
        return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")

    try:
        for _finder, pkg_name, _ispkg in iter_namespace(package_module):
            importlib.import_module(pkg_name)
    except ImportError as e:
        raise PluginLoadError(
            f"Failed to load plugin '{plugin_path}'. Verify the path and installation: {str(e)}"
        ) from e


def get_path_arg(field_name: str, args: Sequence[str] | None = None) -> str | None:
    return parse_arg(f"{field_name}.{PATH_KEY}", args)


def get_type_arg(field_name: str, args: Sequence[str] | None = None) -> str | None:
    return parse_arg(f"{field_name}.{draccus.CHOICE_TYPE_KEY}", args)


def filter_arg(field_to_filter: str, args: Sequence[str] | None = None) -> list[str]:
    if args is None:
        return []
    return [arg for arg in args if not arg.startswith(f"--{field_to_filter}=")]


def filter_path_args(fields_to_filter: str | list[str], args: Sequence[str] | None = None) -> list[str]:
    """
    Filters command-line arguments related to fields with specific path arguments.

    Args:
        fields_to_filter (str | list[str]): A single str or a list of str whose arguments need to be filtered.
        args (Sequence[str] | None): The sequence of command-line arguments to be filtered.
            Defaults to None.

    Returns:
        list[str]: A filtered list of arguments, with arguments related to the specified
        fields removed.

    Raises:
        ArgumentError: If both a path argument (e.g., `--field_name.path`) and a type
            argument (e.g., `--field_name.type`) are specified for the same field.
    """
    if isinstance(fields_to_filter, str):
        fields_to_filter = [fields_to_filter]

    filtered_args = [] if args is None else list(args)

    for field in fields_to_filter:
        if get_path_arg(field, args):
            if get_type_arg(field, args):
                raise ArgumentError(
                    argument=None,
                    message=f"Cannot specify both --{field}.{PATH_KEY} and --{field}.{draccus.CHOICE_TYPE_KEY}",
                )
            filtered_args = [arg for arg in filtered_args if not arg.startswith(f"--{field}.")]

    return filtered_args


def wrap(config_path: Path | None = None) -> Callable[[F], F]:
    """
    HACK: Similar to draccus.wrap but does additional things:
        - Will remove '.path' arguments from CLI in order to process them later on.
        - If a 'config_path' is passed and the main config class has a 'from_pretrained' method, will
          initialize it from there to allow to fetch configs from the hub directly
        - Will load plugins specified in the CLI arguments.
        - NEW: support --yaml_path=... to load a YAML file and convert it to draccus-style CLI overrides.
    """

    def _yaml_to_cli_args(data: Any, prefix: str = "") -> list[str]:
        """
        Flatten nested YAML dict into draccus-style CLI args:
          {"policy": {"dtype": "bfloat16"}} -> ["--policy.dtype=bfloat16"]
        """
        out: list[str] = []
        if data is None:
            return out

        if isinstance(data, dict):
            for k, v in data.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                out.extend(_yaml_to_cli_args(v, key))
            return out

        key = prefix
        v = data

        if isinstance(v, bool):
            out.append(f"--{key}={'true' if v else 'false'}")
        elif isinstance(v, (int, float, str)):
            out.append(f"--{key}={v}")
        elif isinstance(v, (list, tuple)):
            # safest representation for potential list fields
            import json
            out.append(f"--{key}={json.dumps(v)}")
        else:
            out.append(f"--{key}={str(v)}")
        return out

    def _load_yaml_overrides(yaml_path: str) -> list[str]:
        try:
            import yaml  # pip install pyyaml
        except Exception as e:
            raise RuntimeError("Missing dependency for --yaml_path: please `pip install pyyaml`.") from e

        p = Path(yaml_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"--yaml_path not found: {p}")

        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if not isinstance(cfg, dict):
            raise ValueError(f"YAML root must be a mapping/dict, got {type(cfg)}")

        return _yaml_to_cli_args(cfg)

    def _arg_key(arg: str) -> str | None:
        # "--a.b.c=123" -> "a.b.c"
        if not arg.startswith("--") or "=" not in arg:
            return None
        return arg[2:].split("=", 1)[0].strip() or None

    def wrapper_outer(fn: F) -> F:
        @wraps(fn)
        def wrapper_inner(*args: Any, **kwargs: Any) -> Any:
            argspec = inspect.getfullargspec(fn)
            argtype = argspec.annotations[argspec.args[0]]

            # If user passed the config object directly, do nothing special.
            if len(args) > 0 and type(args[0]) is argtype:
                cfg = args[0]
                args = args[1:]
            else:
                # -----------------------------
                # 1) Start from raw CLI args
                # -----------------------------
                cli_args: list[str] = list(sys.argv[1:])

                # -----------------------------
                # 2) Plugin loading (existing behavior)
                # -----------------------------
                plugin_args = parse_plugin_args(PLUGIN_DISCOVERY_SUFFIX, cli_args)
                for plugin_cli_arg, plugin_path in plugin_args.items():
                    try:
                        load_plugin(plugin_path)
                    except PluginLoadError as e:
                        raise PluginLoadError(f"{e}\nFailed plugin CLI Arg: {plugin_cli_arg}") from e
                    cli_args = filter_arg(plugin_cli_arg, cli_args)

                # -----------------------------
                # 3) Filter '.path' args for path fields (existing behavior)
                # -----------------------------
                if has_method(argtype, "__get_path_fields__"):
                    path_fields = argtype.__get_path_fields__()
                    cli_args = filter_path_args(path_fields, cli_args)

                # -----------------------------
                # 4) NEW: YAML injection
                #    - If --yaml_path is present, load YAML -> overrides
                #    - CLI still has higher priority than YAML
                # -----------------------------
                yaml_path_cli = parse_arg("yaml_path", cli_args)
                if yaml_path_cli:
                    # remove --yaml_path itself (draccus doesn't know this field)
                    cli_args = filter_arg("yaml_path", cli_args)

                    # collect existing explicit CLI keys (to preserve CLI priority)
                    cli_keys = {_arg_key(a) for a in cli_args}
                    cli_keys.discard(None)

                    yaml_args = _load_yaml_overrides(yaml_path_cli)

                    # drop YAML keys that are explicitly set in CLI (so CLI wins)
                    filtered_yaml_args = []
                    for ya in yaml_args:
                        k = _arg_key(ya)
                        if k is not None and k in cli_keys:
                            continue
                        filtered_yaml_args.append(ya)

                    # prepend YAML defaults, then explicit CLI overrides
                    cli_args = filtered_yaml_args + cli_args

                    # IMPORTANT: keep sys.argv consistent because some configs re-read sys.argv later
                    sys.argv = [sys.argv[0]] + cli_args

                # -----------------------------
                # 5) Existing: config_path + from_pretrained behavior
                # -----------------------------
                config_path_cli = parse_arg("config_path", cli_args)
                if has_method(argtype, "from_pretrained") and config_path_cli:
                    # keep config_path in sys.argv (for resume logic), but remove it from overrides
                    cli_args_wo_config_path = filter_arg("config_path", cli_args)
                    cfg = argtype.from_pretrained(config_path_cli, cli_args=cli_args_wo_config_path)
                else:
                    cfg = draccus.parse(config_class=argtype, config_path=config_path, args=cli_args)

            response = fn(cfg, *args, **kwargs)
            return response

        return cast(F, wrapper_inner)

    return cast(Callable[[F], F], wrapper_outer)
