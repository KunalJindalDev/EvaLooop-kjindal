"""Dependency-light protocol, checkpointing, and analysis primitives."""
import hashlib
import json
from pathlib import Path

GENERATION = 'Generate Python code for the following task:\n\n{description}\n\nReturn only the code, without explanations or comments.'
SUMMARY = 'Summarize what the following code does in a detailed description:\n\n```\n{code}\n```\n\nReturn only the summary, without explanations or comments.'


def select_ids(ids, n=100):
    """Outcome-independent, stable hash ordering; same subset for every run."""
    if n < 1 or n > len(ids):
        raise ValueError('Invalid subset size')
    return sorted(ids, key=lambda x: hashlib.sha256(('evaloop-a4-2026:' + x).encode()).hexdigest())[:n]


def metrics(tasks, maximum=10):
    if not tasks:
        raise ValueError('Empty run')
    counts = [t['successful_cycles'] for t in tasks]
    if any(not isinstance(x, int) or not 0 <= x <= maximum for x in counts):
        raise ValueError('Invalid sustained-cycle count')
    return {'n_tasks': len(counts),
            'success_rate': sum(x >= 1 for x in counts) / len(counts),
            'asl_score': sum(x*x for x in counts) / (len(counts)*maximum),
            'mean_cycles': sum(counts)/len(counts),
            'pass_at_10': sum(x == maximum for x in counts)/len(counts)}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def evaluate_task(task_id, initial, model, tester, generation, reverse,
                  maximum=10, checkpoint=None, record=None):
    record = record or dict(task_id=task_id, initial_prompt=initial, cycles=[],
                            successful_cycles=0, max_cycles_reached=False)
    if record['cycles'] and not record['cycles'][-1]['success']:
        return record
    current = record['cycles'][-1]['task_2_output'] if record['cycles'] else initial
    for cycle in range(len(record['cycles']) + 1, maximum + 1):
        raw = model.generate(generation.create_prompt(current))
        code = generation.extract_output(raw)
        tested = tester(code, task_id)
        if 'error' in tested:
            # Infrastructure errors cannot silently become scientific failures.
            raise RuntimeError(tested['error'])
        passed = tested['success']
        item = dict(cycle=cycle, task_id=task_id, input_prompt=current,
                    task_1_raw_output=raw, task_1_output=code,
                    test_results=tested, success=passed)
        if passed:
            raw_summary = model.generate(reverse.create_prompt(code))
            current = reverse.postprocess(reverse.extract_output(raw_summary),
                      {'cycle_input': current, 'initial_prompt': initial})
            item.update(task_2_raw_output=raw_summary, task_2_output=current)
            record['successful_cycles'] = cycle
        record['cycles'].append(item)
        record['max_cycles_reached'] = record['successful_cycles'] == maximum
        if checkpoint:
            checkpoint(record)
        if not passed:
            break
    return record
