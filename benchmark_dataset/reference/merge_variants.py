"""Copy 24 original variants into the active benchmark and append their rows."""
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'benchmark_dataset'
COPY=ROOT/'Test_Dataset/Large-scale-or/Large-scale-or-101_冗余列35个 copy.csv'
SOURCE=COPY.with_name('Large-scale-or-101-variants.csv')

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    old=read(COPY);variants=read(SOURCE)
    assert len(old)==12 and len(variants)==24
    assert list(old[0])==list(variants[0])
    backup=ROOT/'benchmark_archive/before_variants_merge'
    backup.mkdir(exist_ok=True)
    if (backup/'questions.csv').exists():assert read(backup/'questions.csv')==old
    else:shutil.copyfile(COPY,backup/'questions.csv')
    source_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if (BASE/'case_manifest.json').exists():
        shutil.copyfile(BASE/'case_manifest.json',backup/'case_manifest.json')
        manifest=json.loads((BASE/'case_manifest.json').read_text())
    else:
        manifest=[]
        for row in old:
            p=ROOT/row['Dataset_address'].splitlines()[0]
            name=p.relative_to(BASE).parts[0]
            cfg={'case':name}
            if name.startswith('RA'):
                cfg['input_style']='snapshot'
                if name=='RA2':cfg.update(input_style='events',scope_column='tenant',scope_value='NORTH',asof='2026-03-12')
                if name=='RA4':cfg.update(input_style='events',scope_column='dealership_id',scope_value='OSLO_NEW_CARS',asof='2026-05-07')
            manifest.append(cfg)
    merged=list(old);report=[]
    for i,r in enumerate(variants,1):
        name=f'Variant{i}';folder=BASE/name/'inputs'
        assert not folder.exists()
        folder.mkdir(parents=True)
        paths=[];files=[]
        for raw in r['Dataset_address'].replace('\\n','\n').splitlines():
            p=ROOT/raw.strip()
            assert p.is_file() and p.parent.name==name,p
            dest=folder/p.name
            assert not dest.exists()
            shutil.copyfile(p,dest)
            assert p.read_bytes()==dest.read_bytes()
            paths.append(str(dest.relative_to(ROOT)))
            files.append({'source':str(p.relative_to(ROOT)),'destination':paths[-1],'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
        row=dict(r);row['Dataset_address']='\n'.join(paths)
        assert all(row[k]==r[k] for k in r if k!='Dataset_address')
        merged.append(row)
        manifest.append({'case':name,'input_style':'original_variant','source_csv':str(SOURCE.relative_to(ROOT)),'source_row':i,'validation_mode':'copy_integrity'})
        report.append({'case':name,'row':12+i,'files':files,'query_and_labels_preserved':True})
    assert merged[:12]==old and len(merged)==36
    assert all(len(v)<32767 for r in merged for v in r.values())
    with (BASE/'questions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(old[0]),quoting=csv.QUOTE_ALL);w.writeheader();w.writerows(merged)
    COPY.write_bytes((BASE/'questions.csv').read_bytes())
    assert read(COPY)==merged
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==source_hash
    (BASE/'case_manifest.json').write_text(json.dumps(manifest,indent=2))
    (BASE/'reference/variants_merge.json').write_text(json.dumps({'source_sha256':source_hash,'total_cases':36,'variants':report},indent=2))
    print(json.dumps({'total_cases':36,'copied_files':sum(len(r['files']) for r in report),'original_12_preserved':True,'source_variants_preserved':True}))

if __name__=='__main__':main()
