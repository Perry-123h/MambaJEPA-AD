"""Retrieve public BankSim data; reject any file with a mismatching checksum."""
from pathlib import Path
import urllib.request,hashlib,argparse
EXPECTED='e37006f76d993bfaec3a02d717b4f0bdc1ebfa5d36449e91fb7c07e225278377'
URL='https://www.kaggle.com/datasets/ealaxi/banksim1'
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--file');a=p.parse_args()
 target=Path(a.file or 'data/raw/bs140513_032310.csv');target.parent.mkdir(parents=True,exist_ok=True)
 if not a.file: raise SystemExit('Download bs140513_032310.csv from '+URL+' and pass --file PATH; Kaggle may require authentication.')
 actual=hashlib.sha256(target.read_bytes()).hexdigest()
 print('SHA256:',actual)
 if actual!=EXPECTED: raise SystemExit('Checksum mismatch. Do not treat this file as the manuscript dataset.')
 print('Verified against manuscript checksum.')
