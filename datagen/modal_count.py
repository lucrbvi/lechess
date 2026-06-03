from pathlib import Path

import modal
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

APP_NAME = 'lechess-count'
STAGING_VOLUME = 'lechess-hf-staging'
VOLUME_MOUNT = '/out'
VOLUME_PATH = Path(VOLUME_MOUNT)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(STAGING_VOLUME)
image = modal.Image.debian_slim(python_version='3.11').pip_install('pyarrow>=22.0.0', 'tqdm>=4.66.3')

def count_rows(path: Path) -> dict[str, int]:
    counts = {'bad_files': 0}

    for split in ('train', 'test'):
        files = sorted((path / split).glob('*.parquet'))
        counts[split] = 0

        for file in tqdm(files, desc=f'{path.name}/{split}', unit='file'):
            try:
                counts[split] += pq.ParquetFile(file).metadata.num_rows
            except pa.ArrowInvalid:
                counts['bad_files'] += 1
                print(f'bad parquet: {file}')

    counts['total'] = counts['train'] + counts['test']
    return counts

@app.function(image=image, volumes={VOLUME_MOUNT: volume}, timeout=60 * 60)
def count_volume() -> dict[str, dict[str, int]]:
    volume.reload()

    return {
        'large': count_rows(VOLUME_PATH / 'games'),
        'fine': count_rows(VOLUME_PATH / 'stockfish'),
    }

@app.local_entrypoint()
def main() -> None:
    counts = count_volume.remote()

    for dataset, values in counts.items():
        print(f'{dataset}: train={values["train"]:,} test={values["test"]:,} total={values["total"]:,}')
