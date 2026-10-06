"""Recompute WL and aggregate variants by the user-supplied questions.csv."""
import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
DEFAULT = Path('/Users/cora/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_yj16fcohm45a22_43e6/msg/file/2026-10/benchmark_dataset 2/questions.csv')
CATEGORIES = ['NRM', 'RA', 'TP', 'AP', 'UFLP', 'Mixture', 'Others']


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions', type=Path, default=DEFAULT)
    args = parser.parse_args()
    mapping = {}
    for row_number, row in enumerate(read(args.questions), 1):
        ids = set(re.findall(r'(?:^|/)Variant(\d+)/', row['Dataset_address']))
        assert len(ids) == 1, (row_number, ids)
        number = int(next(iter(ids)))
        raw = row['Problem Type'].strip()
        category = 'Others' if re.match(r'^Others(?:\s|$)', raw) else raw
        assert category in CATEGORIES and number not in mapping
        label_ids = set(re.findall(r'(?:^|/)Variant(\d+)/', row['Label-model']))
        assert not label_ids or label_ids == ids
        mapping[number] = dict(instance=f'Variants{number}', questions_row=row_number,
                               category=category, original_category=raw)
    assert set(mapping) == set(range(1, 37))
    write('variant_category_mapping.csv', [mapping[i] for i in sorted(mapping)])
    subprocess.run([sys.executable, str(OUT / 'recompute_variants_wl.py')],
                   check=True, capture_output=True, text=True)
    refs = [r for r in read(OUT / 'manifest.csv') if r['dataset'] == 'reference']
    summaries = []
    for h in (1, 2, 3):
        details = []
        for r in read(OUT / f'pairwise_h{h}.csv'):
            number = int(r['instance'].replace('Variants', ''))
            meta = mapping[number]
            same = [x for x in refs if x['category'] == meta['category']]
            assert same
            best_same = max(same, key=lambda x: float(r[x['instance']]))
            best_all = max(refs, key=lambda x: float(r[x['instance']]))
            details.append(dict(**meta, nearest_same=best_same['instance'],
                similarity_same=float(r[best_same['instance']]),
                nearest_all=best_all['instance'], similarity_all=float(r[best_all['instance']])))
        assert len(details) == 36
        assert all(r['similarity_all'] + 1e-12 >= r['similarity_same'] for r in details)
        write(f'classified_nearest_h{h}.csv', details)
        for category in CATEGORIES + ['All']:
            selected = [r for r in details if category == 'All' or r['category'] == category]
            sv = [r['similarity_same'] for r in selected]
            av = [r['similarity_all'] for r in selected]
            summaries.append(dict(h=h, category=category, instances=len(selected),
                median_same=float(np.median(sv)) if sv else None,
                mean_same=float(np.mean(sv)) if sv else None,
                median_all=float(np.median(av)) if av else None,
                mean_all=float(np.mean(av)) if av else None))
    write('category_summary.csv', summaries)
    primary = [r for r in summaries if r['h'] == 2]
    text = ['# 36 个变体按真实类别汇总的 WL 相似度', '',
        '使用用户提供的 questions.csv 中的 Problem Type，按 Dataset_address 中的 Variant 编号匹配 VariantsN.lp，不按 CSV 行顺序匹配。Others 的子类型合并为 Others，原始子类型保存在逐题表。', '',
        '参考为当前 Ref_Data_Large_Scale_LP 的 15 个 LP；参考类别取自 RAG_Examples_All.csv。采用与之前相同的 typed 变量—约束二部图、累计 WL h=2、余弦归一化、无 presolve。先对每题在同类别参考中取最大相似度，再对该类别逐题分数取中位数/均值。全库列对全部 15 个参考取最大值。', '',
        '| 类别 | 测试数 | 同类别最近相似度中位数 | 同类别均值 | 全参考库最近中位数 |',
        '|---|---:|---:|---:|---:|']
    fmt = lambda v: '—' if v is None else f'{v:.4f}'
    for r in primary:
        label = '全部' if r['category'] == 'All' else r['category']
        text.append(f"| {label} | {r['instances']} | {fmt(r['median_same'])} | {fmt(r['mean_same'])} | {fmt(r['median_all'])} |")
    text += ['', 'NRM、RA、TP 若测试数为 0，表示这份 questions.csv 未将任何变体标为这些类别，不是模型读取失败，也不表示相似度为 0。Others 是异质类别池；同类匹配不保证属于同一子类型。所有类别均使用 CSV 原始分类，不因最近参考类别或模型外观重新分类。', '',
        '全部行直接统计 36 个逐题分数，不是类别均值的简单平均。分数越高表示当前表示下越接近；不代表求解准确率或鲁棒性。WL 忽略目标、系数值/符号、RHS 和界值。', '',
        '复现：`/usr/bin/python3 outputs/wl_variants_20261004/summarize_categories.py --questions "/完整路径/questions.csv"`。', '',
        '明细：classified_nearest_h2.csv；完整配对：pairwise_h2.csv；映射：variant_category_mapping.csv；h=1/2/3 汇总：category_summary.csv。']
    (OUT / 'CATEGORY_RESULTS_CN.md').write_text('\n'.join(text) + '\n')
    audit = dict(questions_path=str(args.questions),
        questions_sha256=hashlib.sha256(args.questions.read_bytes()).hexdigest(),
        mapping='Variant number parsed from Dataset_address; not CSV row order',
        category_counts=dict(Counter(r['category'] for r in mapping.values())),
        original_category_counts=dict(Counter(r['original_category'] for r in mapping.values())),
        expected_and_matched_variants=36, primary_h=2,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT / 'category_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(primary, ensure_ascii=False, indent=2))
    print('CATEGORY MEMBERS', {c: [r['instance'] for r in mapping.values() if r['category']==c] for c in CATEGORIES})


if __name__ == '__main__':
    main()
