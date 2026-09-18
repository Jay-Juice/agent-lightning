"""Fetch immutable upstream pydicom data and verify every upstream SHA256."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

OUT = Path(__file__).with_name('pydicom-download-fixtures-v2')
PINNED = {'urls.json': '7d706c53f608f0f31f2074637ce15eb41fb400c112812c5ba14154ef28b5c712',
          'hashes.json': 'e4ad0803a8226ffdb199c73c75244457c0d998e8aab7c90d9c698253758a6ff3'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, digest in PINNED.items():
        path = OUT / name
        if not path.exists():
            url = f'https://raw.githubusercontent.com/pydicom/pydicom/7d361b3d/src/pydicom/data/{name}'
            with urlopen(url, timeout=90) as response:
                content = response.read()
            assert hashlib.sha256(content).hexdigest() == digest, name
            path.write_bytes(content)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    urls = json.loads((OUT / 'urls.json').read_text())
    hashes = {k.lower(): v for k, v in json.loads((OUT / 'hashes.json').read_text()).items()}
    (OUT / 'data').mkdir(exist_ok=True)

    def fetch(item):
        name, url = item
        assert Path(name).name == name and '/' not in name and '\\' not in name
        assert url.startswith('https://github.com/pydicom/pydicom-data/raw/')
        raw = url.replace('https://github.com/pydicom/pydicom-data/raw/',
                          'https://raw.githubusercontent.com/pydicom/pydicom-data/', 1)
        expected = hashes[name.lower()]
        path = OUT / 'data' / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            temporary = path.with_suffix('.download')
            with urlopen(raw, timeout=90) as src, temporary.open('wb') as dst:
                while block := src.read(1024 * 1024):
                    dst.write(block)
            assert hashlib.sha256(temporary.read_bytes()).hexdigest() == expected, name
            temporary.replace(path)
        return {'name': name, 'url': url, 'path': urlsplit(url).path,
                'sha256': expected, 'size': path.stat().st_size}

    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(fetch, x) for x in urls.items()]):
            record = future.result()
            records.append(record)
            print(json.dumps({'completed': len(records), 'total': len(urls),
                              'file': record['name'], 'bytes': record['size']}), flush=True)
    records.sort(key=lambda x: x['name'])
    (OUT / 'manifest.json').write_text(json.dumps({'metadata_sha256': PINNED, 'files': records}, indent=2))
    print('VERIFIED_BYTES', sum(r['size'] for r in records), flush=True)


if __name__ == '__main__':
    main()
