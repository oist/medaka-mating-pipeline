from pathlib import Path

from medaka_pipeline.io.paths import ProjectPaths


def test_project_paths() -> None:
    root = Path("/tmp")
    paths = ProjectPaths(root=root)
    assert paths.configs == root / "configs"
    assert paths.data == root / "data"
    assert paths.runs == root / "runs"
