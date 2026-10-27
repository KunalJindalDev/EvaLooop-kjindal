# Assignment 4: code ↔ pseudocode

Status: all four GPU experiments completed and their result records passed independent validation. The student has confirmed initial inspection of the 20 failure cases and review of the analysis and report. The report remains a working draft for further revision before submission.

Public fork: https://github.com/KunalJindalDev/EvaLooop-kjindal
Upstream: https://github.com/evalooop/EvaLooop at `ce22fe2f80c8f36da85cc8a0cca49c2bcc8b1cb9`.
Paper: https://arxiv.org/abs/2505.12185.

## Research question and protocol

Does replacing prose summaries with explicit pseudocode improve sustained functional correctness, and does the effect differ between Qwen2.5-Coder 0.5B and 1.5B Instruct?

Both configurations start from the same original MBPP Plus description and use the exact generation template in the course package. After each passing generation, the evaluated model alone produces either a prose summary (course template) or structured pseudocode (INPUT, OUTPUT, numbered STEPS). The reverse step sees only the generated code. The next description contains that output and the original final example assertion, wrapped as a docstring. The assertion preserves the called function name and example argument structure; it does not mathematically guarantee a full signature. No original prose, hidden tests, canonical solution, external attack model, repairs, or judge outputs enter the prompt. The pseudocode instruction is fixed across tasks, not tailored after observing failures.

The same generation template accepts the pseudocode as the next task description. Both loops stop at the first failure of either the base or Plus tests, or after 10 successful cycles. Empty/invalid/truncated code is tested as returned. Infrastructure exceptions abort instead of being counted as model errors. A reverse response at cycle 10 is logged for protocol consistency but never used for generation. Nonconforming pseudocode is retained rather than repaired or rejected; format compliance is a manual-analysis topic.

## Models and setup

Use `Qwen/Qwen2.5-Coder-0.5B-Instruct` and `Qwen/Qwen2.5-Coder-1.5B-Instruct`, each with `standard` and `pseudocode`. These smaller models are absent from the course per-cycle package; therefore all four runs are fresh. Model cards: https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct and https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct.

One greedy run per configuration, `do_sample=False` (temperature-zero equivalent), 1024 new tokens per call, float16 without quantization, seed 2026, at most 10 cycles. Models use their native user-only chat template. Run metadata records resolved model commit, GPU, software versions, dataset hash, token counts and elapsed time. No API/judge charge; report actual Colab usage separately if billed. Greedy decoding is not a cross-hardware bitwise guarantee; paired analysis flags differences in cycle-1 output.

Select 100 tasks by sorting all task IDs by SHA256 of `evaloop-a4-2026:` plus the ID, then taking the first 100. This selection ignores outcomes. `results/task_ids.json` and each run's metadata record the exact set. All four runs must use the same set. Do not select tasks based on whether either model passes.

## Run in Colab

1. Upload `Run_Assignment4.ipynb` to https://colab.research.google.com/.
2. Select **Runtime → Change runtime type → T4 GPU** (or another GPU).
3. Use a fresh, disposable runtime. Do not mount Drive, log into services, enable notebook secrets, or upload personal files. Generated programs are untrusted. Tests run through EvalPlus `untrusted_check`, which uses child processes, reliability guards and timeouts, inside this disposable VM. These guards are not a strong OS security boundary; the VM must contain no valuable data or credentials.
4. Run the notebook cells in order. Upload `colab_bundle.zip` when prompted. Installation and offline tests precede the four experiment cells.
5. After EACH experiment cell the notebook offers a results ZIP download. Save it, since Colab can discard its local disk. No Drive mount is needed. The final cell produces `experiment_results.zip` with all results and analysis.
6. If disconnected, start a fresh runtime, upload the source bundle, install, then set `RESTORE=True` in the restore cell and upload the last results ZIP. Run the experiment cells again. Complete runs are skipped; partially complete tasks resume from the latest completed cycle. An interrupted in-flight cycle is repeated, not an additional independent run.
7. Return `experiment_results.zip` for the report and failure analysis. Depending on GPU and survival, these runs can take hours; no fixed runtime is promised.

Command equivalent (in that disposable runtime, from the extracted repository):

```bash
python -m pip install -r assignment4/requirements-colab.txt
python -m assignment4.run --model Qwen/Qwen2.5-Coder-0.5B-Instruct --configuration standard --disposable-colab
python -m assignment4.run --model Qwen/Qwen2.5-Coder-0.5B-Instruct --configuration pseudocode --disposable-colab
python -m assignment4.run --model Qwen/Qwen2.5-Coder-1.5B-Instruct --configuration standard --disposable-colab
python -m assignment4.run --model Qwen/Qwen2.5-Coder-1.5B-Instruct --configuration pseudocode --disposable-colab
python -m assignment4.analyze --results results --output analysis
```

No package installation of the entire upstream framework is needed: run Python from the repository root. The pinned requirements cover this extension. The notebook uses uv 0.6.17 to provision an isolated Python 3.12.9 environment and installs PyTorch 2.6.0 with CUDA 12.4 wheels. This avoids depending on the notebook kernel version or its venv/ensurepip installation. The GPU driver remains provided by Colab. Setup streams subprocess output so failures include their actual diagnostic messages.

## Changes and code understanding

`evaloop/tasks/code_pseudocode.py` is a registered inverse task derived from the existing summarization task. The sole edit to existing Python code imports and exports it in `evaloop/tasks/__init__.py`. `assignment4/protocol.py` supplies a task-by-task loop with cycle checkpoints; `run.py` combines the upstream generation/extraction classes and EvalPlus tester with a small-model Transformers adapter. `analyze.py` validates complete paired runs and calculates metrics. This entry point is `python -m assignment4.run`; the upstream `evaloop evaluate` CLI is not modified to select pseudocode.

Why a new driver: the current upstream `BatchEvaluationCycle` runs both tasks before `EvaLoopEvaluator` tests code, unlike the course package's test-before-summary description. The new driver tests first (for both conditions), avoids spending tokens on summaries of failed code, persists progress, and samples a paired subset. Upstream's current default prompt templates also differ from the supplied package; we explicitly use the package templates for fresh standard runs. We do not silently claim exact reproduction of the historical leaderboard.

Upstream `CodeGenerationTask.create_prompt` formats the task and `extract_output` takes the first fenced block or the entire stripped response. `CodeSummarizationTask` strips the reverse output and appends the assert from the prior description; our pseudocode task obtains the same anchor from the immutable initial description. `EvalPlusCodeTester` runs both original and augmented MBPP tests against expected outputs and marks a pass only if both pass. Expected outputs use the benchmark canonical implementation, never the evaluated model. The results use upstream keys `model`, `max_cycles`, `prompt_results`, `successful_cycles`, `task_1_output`, `task_2_output`, `success`, and `test_results`, with extra metadata and completion flags.

## Reproducing analysis and finishing the report

`python -m assignment4.analyze` refuses partial runs, duplicate IDs, mismatched tasks, and invalid stop/test records. `summary.csv` has exactly the required header. Success rate means pass@cycle-1, not survival through cycle 10. `asl_score = sum(successful_cycles**2)/(n_tasks*10)`, matching the package's `aslScore` / upstream `ASL_pow`. Also report pass@10 and the ordinary mean in `metrics.json`. The upstream headline semantic-weighted `ASL` is different; no judge-weighted score is claimed here.

The analyzer creates paired per-task comparisons and a worksheet containing ten extension failures per model, prioritizing failures after at least one successful cycle. `failure_evidence.json` includes their complete extension and matched baseline records. Read original prompt, last passing code, its pseudocode, and first failing code. Fill `failure_review.csv` yourself with category, concrete evidence, baseline comparison and reviewer. Suggested categories: omitted requirement; return type/order/duplicate drift; inability to implement preserved description; function interface drift; code extraction/truncation; initial-generation error; ambiguous. Separate direct observations from inferred causes. Initial-generation failures cannot be attributed to the reverse task.

Inspect at least 20 actual failed extension loops. If there are fewer, enlarge the common task set for all four runs using a new output directory and explicitly document the extension of the sample. Do not overwrite original runs or cherry-pick successful results. Include one extension-induced failure relative to baseline and one sustained case if observed; otherwise state that the requested type was not observed. Report rankings, effect size, task-level wins/losses/ties and threats to validity. The current working draft is `REPORT_DRAFT.md`; review revisions against the saved evidence and keep the rendered report within three single-column pages.

### AI assistance and responsibility

I used a coding assistant to help implement and debug the extension and prepare preliminary analysis and report text. I ran the experiments in Colab, managed the resulting artifacts, inspected the 20 failure cases, and reviewed the analysis and report. I take responsibility for all answers, interpretations, and conclusions in this submission, including any errors. No AI judge was used to determine test outcomes or calculate ASL.

## Public fork and final submission

The public fork is linked above. The patch applies against the pinned upstream commit and includes the extension, tests and documentation. The submitted result records retain the experimental source fingerprint; documentation updates after the run do not change the experiment code. Do not upload the supplied course dataset or include it in the submission ZIP.

Final ZIP must include report PDF (≤3 pages), README with public fork URL, `extension.patch`, four complete run JSON files, `summary.csv`, and preferably task manifest and completed review worksheet. The source bundle and notebook are preparation artifacts, not a completed submission. Keep raw responses and metadata in the submitted run JSON files.

## Local verification

```bash
PYTHONPATH=. python -m pytest tests/test_assignment4.py -q
```

Tests use fake models/test outcomes; they do not generate experimental findings. They check task registration, immutable assertion retention, outcome-independent sampling, metric normalization, stopping, cycle-10 cap, resume behavior, and infrastructure error handling.

## Completed experiment results

All runs used the same 100 tasks and a Tesla T4. Initial generated outputs match exactly across conditions for both models. Recomputed metrics match the Colab exports.

| Model | Configuration | Success rate | ASL_pow | Pass@10 |
|---|---|---:|---:|---:|
| Qwen2.5-Coder-0.5B-Instruct | standard | 0.45 | 3.652 | 0.36 |
| Qwen2.5-Coder-0.5B-Instruct | pseudocode | 0.45 | 3.123 | 0.31 |
| Qwen2.5-Coder-1.5B-Instruct | standard | 0.51 | 5.001 | 0.50 |
| Qwen2.5-Coder-1.5B-Instruct | pseudocode | 0.51 | 3.717 | 0.37 |

Recorded evaluation time: 3.47 hours excluding setup. Total: 3,424 model calls, 579,790 input tokens and 353,911 output tokens. API charges: $0; any Colab charge is not known from the run records. Pseudocode lowers aggregate ASL for both models but improves some individual tasks. Detailed analysis and raw results accompany the course submission; the course-provided data package is not redistributed in this fork.
