from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import logging
import os
import re
from urllib.request import urlopen

import modal

LOGGER = logging.getLogger(__name__)
APP_NAME = 'lechess-data'
STAGING_VOLUME = 'lechess-hf-staging'
BASE_URL = 'https://database.lichess.org/standard'
LIST_URL = f'{BASE_URL}/list.txt'
VOLUME_MOUNT = '/out'
VOLUME_PATH = Path(VOLUME_MOUNT)
HF_SECRET = 'huggingface-secret'
DEFAULT_CPU = 2
DEFAULT_MEMORY = 4096
DEFAULT_MAX_CONTAINERS = 24
DEFAULT_SHARD_GAMES = 500000
BANNED_MONTHS = {
    (2023, 11),
    (2022, 12),
    (2021, 3),
    (2021, 2),
    (2020, 12),
    (2020, 7),
}
PGN_RE = re.compile(r'lichess_db_standard_rated_(\d{4})-(\d{2})\.pgn\.zst$')

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(STAGING_VOLUME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version='3.11')
    .apt_install('ca-certificates')
    .pip_install_from_pyproject('pyproject.toml')
    .add_local_python_source('data')
)

def lichess_urls() -> list[str]:
    with urlopen(LIST_URL, timeout=30) as response:
        urls = response.read().decode().splitlines()

    selected = []

    for url in urls:
        match = PGN_RE.search(url)

        if not match:
            continue

        year, month = int(match.group(1)), int(match.group(2))

        if 2017 <= year <= 2026 and (year, month) not in BANNED_MONTHS:
            selected.append(url if url.startswith('https://') else f'{BASE_URL}/{url}')

    return selected

@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY,
    timeout=24 * 60 * 60,
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0, initial_delay=30.0),
    max_containers=DEFAULT_MAX_CONTAINERS,
)
def parse_url(url: str, shard_games: int = DEFAULT_SHARD_GAMES, min_elo: int = 2000) -> dict[str, int | float | str]:
    from .data import export_game_shards

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    counts = export_game_shards(
        [url],
        out_dir=VOLUME_PATH / 'games',
        shard_games=shard_games,
        stockfish_out_dir=VOLUME_PATH / 'stockfish',
        min_elo=min_elo,
        workers=1,
        writer_threads=2,
        parse_threads=DEFAULT_CPU,
    )
    volume.commit()

    return {'url': url, **counts}

@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    secrets=[modal.Secret.from_name(HF_SECRET)],
    cpu=4,
    memory=8192,
    timeout=24 * 60 * 60,
)
def push_to_hub(games_repo: str = '', stockfish_repo: str = '', private: bool = False, revision: str = 'main') -> None:
    from .data import push_datasets_to_hub

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    if games_repo:
        os.environ['LECHESS_HF_GAMES_REPO'] = games_repo

    if stockfish_repo:
        os.environ['LECHESS_HF_STOCKFISH_REPO'] = stockfish_repo

    volume.reload()
    push_datasets_to_hub(VOLUME_PATH / 'games', VOLUME_PATH / 'stockfish', private=private, revision=revision)

def totals(results: Iterable[dict[str, int | float | str]]) -> dict[str, int | float]:
    out: dict[str, int | float] = {'train': 0, 'test': 0, 'stockfish_train': 0, 'stockfish_test': 0, 'seconds': 0.0}

    for result in results:
        for key in out:
            value = result[key]

            if isinstance(value, int | float):
                out[key] += value

    return out

@app.local_entrypoint()
def main(limit: int = 0, shard_games: int = DEFAULT_SHARD_GAMES, min_elo: int = 2000,
         push: bool = False, private: bool = False, revision: str = 'main',
         games_repo: str = '', stockfish_repo: str = '', oldest: bool = False, push_only: bool = False) -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    if push_only:
        push_to_hub.remote(games_repo=games_repo, stockfish_repo=stockfish_repo, private=private, revision=revision)
        return

    urls = lichess_urls()

    if oldest:
        urls.reverse()

    if limit > 0:
        urls = urls[:limit]

    LOGGER.info('selected %d lichess dumps', len(urls))
    results = list(parse_url.map(urls, kwargs={'shard_games': shard_games, 'min_elo': min_elo}))
    LOGGER.info('done %s', totals(results))

    if push:
        push_to_hub.remote(games_repo=games_repo, stockfish_repo=stockfish_repo, private=private, revision=revision)
