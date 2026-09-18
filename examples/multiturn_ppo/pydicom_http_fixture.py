"""Serve pinned public pydicom downloads inside an otherwise offline container.

The original HTTPS URLs, download implementation, tests, hashes, and simulated
network-failure paths are unchanged. Trust is scoped to this grading process.
"""
import hashlib
import io
import json
from pathlib import Path
import tarfile
from urllib.parse import urlsplit

METADATA = {
    "urls.json": "7d706c53f608f0f31f2074637ce15eb41fb400c112812c5ba14154ef28b5c712",
    "hashes.json": "e4ad0803a8226ffdb199c73c75244457c0d998e8aab7c90d9c698253758a6ff3",
}
FIXTURE_DIR = Path('/media/ubuntu/D1/zsj/agent-lightning-runtime/data/swe-smith-training/pydicom-download-fixtures-v2')
SERVER = '''import json, ssl, shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
root=Path('/root/agl-pydicom-downloads')
routes={record['path']: root/'data'/record['name'] for record in json.loads((root/'manifest.json').read_text())['files']}
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.headers.get('Host','').split(':')[0] != 'github.com' or self.path not in routes:
            self.send_error(404); return
        path=routes[self.path]
        self.send_response(200)
        self.send_header('Content-Type','application/octet-stream')
        self.send_header('Content-Length',str(path.stat().st_size))
        self.end_headers()
        with path.open('rb') as stream: shutil.copyfileobj(stream,self.wfile)
    def log_message(self,*args): pass
server=ThreadingHTTPServer(('127.0.0.1',443),Handler)
context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain('/tmp/agl-pydicom-ca.pem','/root/agl-pydicom-key.pem')
server.socket=context.wrap_socket(server.socket,server_side=True)
server.serve_forever()
'''


def verified_files(directory):
    directory = Path(directory)
    for name, expected in METADATA.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Official pydicom metadata changed: {name}')
    urls = json.loads((directory / 'urls.json').read_text())
    hashes = {k.lower(): v for k, v in json.loads((directory / 'hashes.json').read_text()).items()}
    manifest = json.loads((directory / 'manifest.json').read_text())
    records = manifest['files']
    if len(records) != len(urls) or {r['name'] for r in records} != set(urls):
        raise RuntimeError('Incomplete or duplicate pydicom downloads')
    for record in records:
        name = record['name']
        if Path(name).name != name or '/' in name or '\\' in name:
            raise RuntimeError('Invalid fixture filename')
        url = urls[name]
        if (urlsplit(url).scheme != 'https' or urlsplit(url).netloc != 'github.com'
                or record['url'] != url or record['path'] != urlsplit(url).path
                or record['sha256'] != hashes[name.lower()]):
            raise RuntimeError('Fixture is not the pinned official download')
        body = (directory / 'data' / name).read_bytes()
        if len(body) != record['size'] or hashlib.sha256(body).hexdigest() != record['sha256']:
            raise RuntimeError(f'Official pydicom test-data checksum differs: {name}')
    return manifest


def install_download_fixture(box, nodes, directory=FIXTURE_DIR):
    if ('tests/test_data_manager.py::test_fetch_data_files' not in nodes
            or not box.container.labels['agl.instance_id'].startswith('pydicom__pydicom.7d361b3d.')):
        return None
    directory = Path(directory)
    manifest = verified_files(directory)
    # Only support the observed image revision; never overwrite its URL/hash
    # metadata or a candidate's edits to those source-controlled files.
    for name, expected in METADATA.items():
        content = box.root(['cat', f'src/pydicom/data/{name}']).encode()
        if hashlib.sha256(content).hexdigest() != expected:
            raise RuntimeError(f'Image/candidate pydicom metadata differs: {name}')
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w') as tar:
        files = [('manifest.json', (directory / 'manifest.json').read_bytes()),
                 ('server.py', SERVER.encode())]
        files.extend((f"data/{r['name']}", (directory / 'data' / r['name']).read_bytes()) for r in manifest['files'])
        for name, content in files:
            info = tarfile.TarInfo('agl-pydicom-downloads/' + name)
            info.size, info.mode = len(content), 0o444
            tar.addfile(info, io.BytesIO(content))
    if not box.container.put_archive('/root', buf.getvalue()):
        raise RuntimeError('Could not stage official pydicom downloads')
    box.root(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '2',
              '-subj', '/CN=github.com', '-addext', 'subjectAltName=DNS:github.com',
              '-keyout', '/root/agl-pydicom-key.pem', '-out', '/tmp/agl-pydicom-ca.pem'])
    box.root(['chmod', '444', '/tmp/agl-pydicom-ca.pem'])
    box.root(['/opt/miniconda3/envs/testbed/bin/python', '-c',
              "from pathlib import Path; p=Path('/etc/hosts'); "
              "p.write_text(p.read_text()+'\\n127.0.0.1 github.com\\n')"])
    box.container.exec_run(['/opt/miniconda3/envs/testbed/bin/python',
                            '/root/agl-pydicom-downloads/server.py'], user='0:0', detach=True)
    box.root(['/opt/miniconda3/envs/testbed/bin/python', '-c',
              "import socket,time\nfor i in range(50):\n"
              " try:\n  socket.create_connection(('127.0.0.1',443),timeout=.1).close(); break\n"
              " except OSError: time.sleep(.1)\nelse: raise RuntimeError('Fixture HTTPS server failed')"])
    return {
        'environment': {'SSL_CERT_FILE': '/tmp/agl-pydicom-ca.pem',
                        'REQUESTS_CA_BUNDLE': '/tmp/agl-pydicom-ca.pem',
                        'NO_PROXY': 'github.com,127.0.0.1,localhost', 'no_proxy': 'github.com,127.0.0.1,localhost'},
        'provenance': {'files': len(manifest['files']), 'total_bytes': sum(r['size'] for r in manifest['files']),
                       'manifest_sha256': hashlib.sha256((directory / 'manifest.json').read_bytes()).hexdigest(),
                       'metadata_sha256': METADATA, 'transport': 'loopback_https', 'external_network': False},
    }
