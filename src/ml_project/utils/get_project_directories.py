from pathlib import Path

def _find_project_root(start: Path, to_find: str = ".git") -> Path:

    start = start.resolve()
    for cur in (start, *start.parents):
        if (cur / to_find).exists():
            return cur

    raise RuntimeError(
        f"Unable to find the following {to_find} to determine the root in the project."
    )

def get_project_directories():
    root_dir = _find_project_root(Path(__file__).parent)

    paths = {
        "root": root_dir,
        "src": root_dir / "src",
        "notebooks": root_dir / "notebooks",
        "data": root_dir / "data",
        "data_raw": root_dir / "data" / "raw",
        "data_interim": root_dir / "data" / "interim",
        "data_processed": root_dir / "data" / "processed",
        "models": root_dir / "models",
        "reports": root_dir / "reports",
        "figures": root_dir / "reports" / "figures",
        "env": root_dir
    }

    return paths
