"""Authenticate archival bytes only. No scientific admission. Apache-2.0."""
from pathlib import Path, PurePosixPath
import argparse,base64,gzip,hashlib,io,json,tarfile
P=Path(__file__).resolve().parent
def need(ok,msg):
    if not ok: raise ValueError(msg)
def digest(b):return hashlib.sha256(b).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--extract',type=Path);args=parser.parse_args()
    m=json.loads((P/'MANIFEST.json').read_bytes());b=(P/'snapshot.tar.gz.b64').read_bytes()
    need(digest(b)==m['encoded_sha256'],'Encoded archive hash')
    packed=base64.b64decode(b);need(digest(packed)==m['archive_sha256'],'Archive hash')
    expected={r['path']:r for r in m['files']};payload={}
    with tarfile.open(fileobj=io.BytesIO(packed),mode='r:gz') as tar:
        for item in tar.getmembers():
            path=PurePosixPath(item.name)
            need(item.isfile() and not path.is_absolute() and '..' not in path.parts,'Unsafe member')
            need(item.name not in payload and item.name in expected,'Unexpected or duplicate member')
            raw=tar.extractfile(item).read();row=expected[item.name]
            need(len(raw)==row['bytes'] and digest(raw)==row['public_sha256'],'Member hash '+item.name)
            payload[item.name]=raw
    need(set(payload)==set(expected),'Missing members')
    word=json.loads(gzip.decompress(payload['word-h5b-best.json.gz']))
    need(digest(json.dumps(word,sort_keys=True,separators=(',',':')).encode())==m['candidate_canonical'],'Candidate canonical identity')
    if args.extract:
        need(not args.extract.exists(),'Use a new extraction directory')
        args.extract.mkdir(parents=True)
        for name,raw in payload.items():
            out=args.extract/name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(raw)
    print('PASS_ARCHIVAL_INTEGRITY_ONLY',len(payload),'files; no mathematical validation')
if __name__=='__main__':main()
