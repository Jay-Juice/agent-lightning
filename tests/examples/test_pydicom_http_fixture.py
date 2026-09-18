import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def fixture_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / 'examples/multiturn_ppo'))
    return importlib.import_module('pydicom_http_fixture')


def tiny_fixture(tmp_path, monkeypatch, module):
    data = b'official fixture bytes'
    digest = hashlib.sha256(data).hexdigest()
    url = 'https://github.com/pydicom/pydicom-data/raw/pinned/data/a.dcm'
    urls = json.dumps({'a.dcm': url}).encode()
    hashes = json.dumps({'a.dcm': digest}).encode()
    (tmp_path / 'urls.json').write_bytes(urls)
    (tmp_path / 'hashes.json').write_bytes(hashes)
    monkeypatch.setattr(module, 'METADATA', {'urls.json': hashlib.sha256(urls).hexdigest(),
                                           'hashes.json': hashlib.sha256(hashes).hexdigest()})
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data/a.dcm').write_bytes(data)
    manifest = {'files': [{'name': 'a.dcm', 'url': url, 'path': '/pydicom/pydicom-data/raw/pinned/data/a.dcm',
                           'sha256': digest, 'size': len(data)}]}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    return manifest


def test_complete_pinned_download_set(fixture_module, monkeypatch, tmp_path):
    manifest = tiny_fixture(tmp_path, monkeypatch, fixture_module)
    assert fixture_module.verified_files(tmp_path) == manifest


@pytest.mark.parametrize('damage', ['bytes', 'metadata', 'duplicate', 'route'])
def test_fixture_tampering_rejected(fixture_module, monkeypatch, tmp_path, damage):
    manifest = tiny_fixture(tmp_path, monkeypatch, fixture_module)
    if damage == 'bytes':
        (tmp_path / 'data/a.dcm').write_bytes(b'wrong')
    elif damage == 'metadata':
        (tmp_path / 'urls.json').write_bytes(b'{}')
    elif damage == 'duplicate':
        manifest['files'] *= 2
    else:
        manifest['files'][0]['path'] = '/unexpected/path'
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError):
        fixture_module.verified_files(tmp_path)


def test_unrelated_tasks_never_install_service(fixture_module):
    box = SimpleNamespace(container=SimpleNamespace(labels={'agl.instance_id': 'another__project.task'}))
    assert fixture_module.install_download_fixture(box, ['tests/test_data_manager.py::test_fetch_data_files']) is None
    assert fixture_module.install_download_fixture(box, []) is None
