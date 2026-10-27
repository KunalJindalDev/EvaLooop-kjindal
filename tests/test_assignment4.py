import pytest
from assignment4.protocol import evaluate_task, metrics, select_ids, GENERATION, SUMMARY
from evaloop.tasks import CodeGenerationTask, CodeSummarizationTask, CodePseudocodeTask, TaskFactory

INITIAL = '\"\"\"\nDouble a number.\nassert double(2) == 4\n\"\"\"\n'

class Model:
    def __init__(self):
        self.calls = []
    def generate(self, prompt):
        self.calls.append(prompt)
        return 'INPUT n; OUTPUT 2*n; STEPS multiply n by 2' if 'pseudocode' in prompt else 'def double(n): return 2*n'


def test_registration_and_anchor():
    task = TaskFactory().create_task({'type':'code_pseudocode'})
    text = task.postprocess('steps', {'initial_prompt': INITIAL, 'cycle_input': 'bad anchor'})
    assert text.splitlines()[-2] == 'assert double(2) == 4'
    assert 'Double a number' not in task.create_prompt('def double(n): return n*2')
    with pytest.raises(ValueError):
        task.postprocess('steps', {'initial_prompt': 'no assert'})


def test_metrics_and_subset():
    actual = metrics([{'successful_cycles':i} for i in [0,1,2,10]])
    assert actual['asl_score'] == 105/40
    assert actual['success_rate'] == .75
    assert actual['mean_cycles'] == 3.25
    assert select_ids(['a','b','c'], 2) == select_ids(['c','a','b'], 2)


def test_stop_before_summarizing_failure_and_resume():
    m = Model()
    gen, rev = CodeGenerationTask({'prompt_template':GENERATION}), CodePseudocodeTask({})
    outcomes = iter([True, True, False])
    r = evaluate_task('t', INITIAL, m, lambda *_: {'success':next(outcomes)}, gen, rev)
    assert r['successful_cycles'] == 2
    assert len(m.calls) == 5
    assert len(r['cycles']) == 3
    assert 'task_2_output' not in r['cycles'][-1]
    evaluate_task('t', INITIAL, m, lambda *_: pytest.fail('retested failure'), gen, rev, record=r)
    assert len(m.calls) == 5


def test_resume_and_cap():
    m = Model()
    gen, rev = CodeGenerationTask({}), CodePseudocodeTask({})
    saved = []
    def checkpoint(record):
        import copy
        saved.append(copy.deepcopy(record))
        if len(saved) == 2:
            raise InterruptedError()
    with pytest.raises(InterruptedError):
        evaluate_task('t', INITIAL, m, lambda *_: {'success':True}, gen, rev, checkpoint=checkpoint)
    r = evaluate_task('t', INITIAL, m, lambda *_: {'success':True}, gen, rev, record=saved[-1])
    assert r['successful_cycles'] == 10 and len(r['cycles']) == 10
    assert r['max_cycles_reached']
    assert len(m.calls) == 20


def test_infrastructure_error_is_not_model_failure():
    with pytest.raises(RuntimeError, match='missing oracle'):
        evaluate_task('t', INITIAL, Model(), lambda *_: {'error':'missing oracle'},
                      CodeGenerationTask({}), CodePseudocodeTask({}))


def test_standard_prompt_matches_package():
    task = CodeSummarizationTask({'prompt_template': SUMMARY})
    text = task.postprocess('summary', {'cycle_input':INITIAL})
    assert text.splitlines()[-2] == 'assert double(2) == 4'


def test_analysis_rejects_partial_and_bad_stop():
    from assignment4.analyze import validate
    import copy
    tasks = []
    for i in range(100):
        tasks.append({'task_id':str(i), 'successful_cycles':0, 'max_cycles_reached':False,
                      'cycles':[{'cycle':1,'success':False,
                                 'test_results':{'base_success':False,'plus_success':False}}]})
    run = {'complete':True, 'prompt_results':tasks, 'metadata':{'task_ids':[str(i) for i in range(100)]}}
    assert len(validate(run)) == 100
    bad = copy.deepcopy(run)
    bad['complete'] = False
    with pytest.raises(ValueError):
        validate(bad)
    bad = copy.deepcopy(run)
    bad['prompt_results'][0]['cycles'].append({'cycle':2,'success':True})
    with pytest.raises(ValueError):
        validate(bad)


def test_analysis_end_to_end(tmp_path, monkeypatch):
    import json
    import csv
    import sys
    from assignment4.analyze import main
    results, output = tmp_path/'results', tmp_path/'analysis'
    results.mkdir()
    ids = [str(i) for i in range(100)]
    for model in ['small', 'large']:
        for config in ['standard', 'pseudocode']:
            tasks = [dict(task_id=i, initial_prompt=INITIAL, successful_cycles=0,
                          max_cycles_reached=False,
                          cycles=[dict(cycle=1, success=False, task_1_raw_output='invalid',
                                       test_results=dict(base_success=False, plus_success=False))]) for i in ids]
            meta = dict(task_ids=ids, model_revision='commit', dataset_hash='dataset',
                        precision='float16', max_new_tokens=1024, source_sha256='source', packages={})
            run = dict(model=model, configuration=config, complete=True,
                       max_cycles=10, prompt_results=tasks, metadata=meta)
            (results/f'{model}-{config}.json').write_text(json.dumps(run))
    monkeypatch.setattr(sys, 'argv', ['analyze', '--results', str(results), '--output', str(output)])
    main()
    with (output/'summary.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 and all(float(r['asl_score']) == 0 for r in rows)
    with (output/'failure_review.csv').open() as f:
        assert len(list(csv.DictReader(f))) == 20
    assert all(not x['cycle1_output_mismatches'] for x in json.loads((output/'paired_comparison.json').read_text()))
