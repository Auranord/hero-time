from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

from src.ingest.probe import probe_media

app = typer.Typer(help="Twitch VOD highlight pipeline (MVP scaffold).")


@app.command()
def plan(config_path: str = "configs/default.yaml") -> None:
    """Print configured project architecture and stage plan."""

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    typer.echo("Pipeline architecture plan")
    typer.echo(json.dumps(config, indent=2))


@app.command()
def probe(vod_path: str) -> None:
    """Run ingest probe placeholder and print result JSON."""

    result = probe_media(vod_path)
    typer.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()
