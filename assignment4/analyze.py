"""Summarize complete runs; produce a reproducible failure-review worksheet."""
import argparse
import csv
import json
from pathlib import Path
from assignment4.protocol import metrics, save


def validate(run):
    tasks = run['prompt_results']
    ids = [t['task_id'] for t in tasks]
    if not run.get('complete') or len(tasks) < 100 or len(set(ids)) != len(ids):
        raise ValueError('Run is incomplete, too small, or has duplicate tasks')
    if ids != run['metadata']['task_ids']:
        raise ValueError('Task manifest mismatch')
    for t in tasks:
        c = t['cycles']
        if not c or [x['cycle'] for x in c] != list(range(1, len(c)+1)):
            raise ValueError('Nonconsecutive cycle records')
        sustained = sum(x['success'] for x in c)
        if sustained != t['successful_cycles'] or any(not x['success'] for x in c[:-1]):
            raise ValueError('Invalid stopping behavior')
        if len(c) != min(sustained+1, 10) or t['max_cycles_reached'] != (sustained == 10):
            raise ValueError('Missing final cycle')
        for x in c:
            tested = x['test_results']
            if x['success'] != (tested['base_success'] and tested['plus_success']):
                raise ValueError('Test outcome mismatch')
    return tasks


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='results')
    p.add_argument('--output', default='analysis')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for path in sorted(Path(args.results).glob('*.json')):
        value = json.loads(path.read_text())
        if isinstance(value, dict) and 'prompt_results' in value:
            validate(value)
            runs.append(value)
    keys = [(r['model'], r['configuration']) for r in runs]
    models = sorted({x[0] for x in keys})
    if len(models) < 2 or len(keys) != len(set(keys)):
        raise ValueError('Need at least two models and unique runs')
    if set(keys) != {(m,c) for m in models for c in ['standard','pseudocode']}:
        raise ValueError('Each model needs both configurations')
    first_ids = runs[0]['metadata']['task_ids']
    for r in runs:
        if r['metadata']['task_ids'] != first_ids:
            raise ValueError('Unpaired task subsets')
    rows, details = [], []
    for r in runs:
        scores = metrics(r['prompt_results'])
        rows.append(dict(model=r['model'], configuration=r['configuration'],
                         **{k:scores[k] for k in ['n_tasks','success_rate','asl_score']}))
        details.append(dict(model=r['model'], configuration=r['configuration'], **scores))
    with (out/'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['model','configuration','n_tasks','success_rate','asl_score'])
        writer.writeheader()
        writer.writerows(rows)
    save(out/'metrics.json', details)
    comparison = []
    for m in models:
        base, ext = [next(r for r in runs if r['model']==m and r['configuration']==c)
                     for c in ['standard', 'pseudocode']]
        if any(base['metadata'][k] != ext['metadata'][k] for k in
               ['model_revision','dataset_hash','precision','max_new_tokens','source_sha256','packages']):
            raise ValueError('Confounded baseline/extension settings')
        b = {t['task_id']:t for t in base['prompt_results']}
        e = {t['task_id']:t for t in ext['prompt_results']}
        mismatch = [i for i in b if b[i]['cycles'][0]['task_1_raw_output'] != e[i]['cycles'][0]['task_1_raw_output']]
        comparison.append(dict(model=m, cycle1_output_mismatches=mismatch,
                               extension_better=sum(e[i]['successful_cycles'] > b[i]['successful_cycles'] for i in b),
                               extension_worse=sum(e[i]['successful_cycles'] < b[i]['successful_cycles'] for i in b),
                               tied=sum(e[i]['successful_cycles'] == b[i]['successful_cycles'] for i in b)))
    save(out/'paired_comparison.json', comparison)
    # Review ten extension failures per model if possible, prioritizing failures
    # after a successful first cycle. Fill remaining quota with cycle-1 failures.
    review, evidence = [], []
    for m in models:
        ext = next(r for r in runs if r['model']==m and r['configuration']=='pseudocode')
        base = next(r for r in runs if r['model']==m and r['configuration']=='standard')
        b = {t['task_id']:t for t in base['prompt_results']}
        failed = [t for t in ext['prompt_results'] if t['successful_cycles'] < 10]
        failed.sort(key=lambda t: (t['successful_cycles']==0, first_ids.index(t['task_id'])))
        for t in failed[:max(10, 20//len(models))]:
            tid = t['task_id']
            review.append(dict(model=m, task_id=tid, failure_cycle=t['successful_cycles']+1,
                               baseline_successful_cycles=b[tid]['successful_cycles'],
                               category='', evidence='', baseline_comparison='', reviewer=''))
            evidence.append(dict(model=m, task_id=tid, extension=t, baseline=b[tid]))
    # Never overwrite manual annotations.
    review_path = out/'failure_review.csv'
    if not review_path.exists():
        with review_path.open('w') as f:
            w = csv.DictWriter(f, fieldnames=['model','task_id','failure_cycle','baseline_successful_cycles',
                                              'category','evidence','baseline_comparison','reviewer'])
            w.writeheader()
            w.writerows(review)
    save(out/'failure_evidence.json', evidence)
    text = ['# Results pending human interpretation', '', '| Model | Loop | N | Success | ASL_pow |',
            '|---|---|---:|---:|---:|']
    for r in rows:
        text.append(f"| {r['model']} | {r['configuration']} | {r['n_tasks']} | {r['success_rate']:.3f} | {r['asl_score']:.3f} |")
    text += ['', 'Review failure_review.csv against failure_evidence.json. Blank labels are not completed manual analysis.',
             'If fewer than 20 failures are available, expand the task subset and rerun both conditions.']
    (out/'results.md').write_text('\n'.join(text)+'\n')
    print('Wrote', out, '; review cases:', len(review))


if __name__ == '__main__':
    main()
