import os
from pathlib import Path


DEFAULT_ENV_PATH = Path(".env")


def _strip_quotes(value: str):
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]

    return value


def load_env_file(path: str | Path = DEFAULT_ENV_PATH, override: bool = False):
    env_path = Path(path)

    if not env_path.exists():
        return {}

    loaded = {}

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export "):].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_quotes(value.strip())

        if not key:
            continue

        loaded[key] = value

        if override or key not in os.environ:
            os.environ[key] = value

    return loaded
