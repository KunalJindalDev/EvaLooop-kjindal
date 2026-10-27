"""GPU runner. Run ONLY in a disposable Colab VM without Drive or credentials."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

from assignment4.protocol import GENERATION, SUMMARY, evaluate_task, save, select_ids
from evaloop.tasks import CodeGenerationTask, CodeSummarizationTask, CodePseudocodeTask


class HFModel:
    def __init__(self, name, revision):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        if not torch.cuda.is_available():
            raise RuntimeError('Select a GPU runtime in Colab first')
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, revision=revision, torch_dtype=torch.float16, device_map='auto')
        self.model.eval()
        self.tokens = {'input': 0, 'output': 0, 'calls': 0}

    def generate(self, prompt):
        text = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors='pt').to(self.model.device)
        length = inputs['input_ids'].shape[1]
        if length + 1024 > self.model.config.max_position_embeddings:
            raise RuntimeError('Context limit exceeded; no silent truncation allowed')
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=1024,
                                         pad_token_id=self.tokenizer.eos_token_id)
        generated = output[0, length:]
        self.tokens['input'] += length
        self.tokens['output'] += len(generated)
        self.tokens['calls'] += 1
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['Qwen/Qwen2.5-Coder-0.5B-Instruct',
                                      'Qwen/Qwen2.5-Coder-1.5B-Instruct'], required=True)
    p.add_argument('--configuration', choices=['standard', 'pseudocode'], required=True)
    p.add_argument('--output', default='results')
    p.add_argument('--n-tasks', type=int, default=100)
    p.add_argument('--disposable-colab', action='store_true')
    args = p.parse_args()
    if not args.disposable_colab or not Path('/content').is_dir():
        p.error('This runner requires a disposable Colab runtime; see README safety setup')
    if Path('/content/drive').exists():
        p.error('Disconnect Google Drive and factory-reset the runtime before evaluation')
    if args.n_tasks < 100:
        p.error('Assignment runs require at least 100 tasks')
    # EvalPlus uses subprocess isolation, reliability guards and timeouts.
    # No personal files, mounted storage, or secrets may be present in this VM.
    from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash
    from evaloop.evaluation.py_testing import get_groundtruth, EvalPlusCodeTester
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    from huggingface_hub import model_info
    import torch
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    problems = get_mbpp_plus()
    ids = select_ids(list(problems), args.n_tasks)
    selected = {i: problems[i] for i in ids}
    dataset_hash = get_mbpp_plus_hash()
    subset_hash = hashlib.sha256((dataset_hash + json.dumps(ids)).encode()).hexdigest()
    out = Path(args.output)
    path = out / (args.model.split('/')[-1] + '-' + args.configuration + '.json')
    result = json.loads(path.read_text()) if path.exists() else None
    revision_path = out / 'model_revisions.json'
    revisions = json.loads(revision_path.read_text()) if revision_path.exists() else {}
    revision = (result['metadata']['model_revision'] if result else
                revisions.get(args.model) or model_info(args.model).sha)
    if args.model in revisions and revisions[args.model] != revision:
        raise RuntimeError('Model revision differs from paired run')
    revisions[args.model] = revision
    save(revision_path, revisions)
    source_hash = hashlib.sha256()
    for source in ['assignment4/run.py', 'assignment4/protocol.py',
                   'evaloop/tasks/code_pseudocode.py', 'evaloop/tasks/code_generation.py',
                   'evaloop/tasks/code_summarization.py', 'evaloop/evaluation/py_testing.py']:
        source_hash.update(Path(source).read_bytes())
    identity = dict(model_revision=revision, dataset_hash=dataset_hash, task_ids=ids,
                    temperature=0, do_sample=False, max_new_tokens=1024, seed=2026,
                    precision='float16', max_cycles=10, source_sha256=source_hash.hexdigest())
    if result:
        if result['model'] != args.model or result['configuration'] != args.configuration:
            raise RuntimeError('Checkpoint configuration mismatch')
        if any(result['metadata'].get(k) != v for k, v in identity.items()):
            raise RuntimeError('Checkpoint protocol mismatch')
        if result.get('complete'):
            print('Already complete:', path)
            return
    else:
        result = dict(model=args.model, configuration=args.configuration, max_cycles=10,
                      prompt_results=[], complete=False, metadata=identity)
        result['metadata']['packages'] = {n: importlib.metadata.version(n) for n in
                                          ['torch', 'transformers', 'evalplus', 'accelerate']}
        result['metadata']['upstream_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip() if Path('.git').exists() else 'see UPSTREAM_COMMIT'
        result['metadata']['gpu'] = torch.cuda.get_device_name(0)
        result['metadata']['api_cost_usd'] = 0
    save(out / 'task_ids.json', ids)
    save(path, result)
    ground = get_groundtruth(selected, subset_hash, MBPP_OUTPUT_NOT_NONE_TASKS)
    tester = EvalPlusCodeTester(selected, ground)
    model = HFModel(args.model, revision)
    generation = CodeGenerationTask({'prompt_template': GENERATION})
    reverse = (CodePseudocodeTask({}) if args.configuration == 'pseudocode'
               else CodeSummarizationTask({'prompt_template': SUMMARY}))
    start = time.time()
    previous_seconds = result['metadata'].get('elapsed_seconds', 0)
    previous_tokens = result['metadata'].get('tokens', {'input': 0, 'output': 0, 'calls': 0})

    def checkpoint(record):
        existing = next((i for i, t in enumerate(result['prompt_results'])
                         if t['task_id'] == record['task_id']), None)
        if existing is None:
            result['prompt_results'].append(record)
        else:
            result['prompt_results'][existing] = record
        result['metadata']['elapsed_seconds'] = previous_seconds + time.time() - start
        result['metadata']['tokens'] = {k: previous_tokens[k] + v for k, v in model.tokens.items()}
        save(path, result)

    def test(code, tid):
        value = tester.test_code(code, tid, selected[tid]['entry_point'])
        for key in ['base_details', 'plus_details']:
            if key in value:
                value[key] = [bool(x) for x in value[key]]
        return value

    for index, tid in enumerate(ids):
        record = next((t for t in result['prompt_results'] if t['task_id'] == tid), None)
        evaluate_task(tid, selected[tid]['prompt'], model, test, generation, reverse,
                      checkpoint=checkpoint, record=record)
        print(f'{index+1}/{len(ids)} {tid}', flush=True)
    result['complete'] = True
    result['average_successful_cycles'] = sum(t['successful_cycles'] for t in result['prompt_results']) / len(ids)
    save(path, result)
    print('Completed', path)


if __name__ == '__main__':
    main()
