# Does structured pseudocode preserve code-model robustness?
**Evidence-based draft — student confirmation of the 20-case review is pending.**

## 1. Research question and extension
Does replacing prose summaries with structured pseudocode improve sustained functional correctness, and does the effect differ between Qwen2.5-Coder 0.5B and 1.5B Instruct? The expectation was that explicit algorithmic steps might prevent omissions. The observed aggregate effect was the opposite: unweighted ASL declined for both models, particularly 1.5B, although some individual tasks improved.

Option B adds code ↔ pseudocode. The same evaluated model translates its passing code into INPUT, OUTPUT and numbered STEPS. A fixed prompt asks it to preserve actual conditions, loop bounds, ordering, duplicates, return types and exceptional behavior, without repairs or added requirements. The reverse step sees only generated code. Its output plus the original example assertion becomes the next generation description. No external attacker, judge, reference solution or hidden tests enter model prompts. Nonconforming pseudocode is retained; no format repair is performed.

## 2. Framework and changes
EvaLoop starts from an MBPP Plus description containing an example assertion. CodeGenerationTask.create_prompt formats the request; extract_output selects the first fenced block, or the full stripped response if no block exists. EvalPlusCodeTester compares the generated program with benchmark-derived expected outputs on both original and augmented tests. A cycle passes only when both suites pass. After a pass, the same model produces a reverse description, wrapped as a docstring with the original assertion appended. That description is the next input. The first failing generation ends the loop; ten passing cycles is the cap. The assertion anchors the expected function name and example arguments, while revealing one example. It does not guarantee an unchanged interface or full semantics.

The extension adds registered CodePseudocodeTask and its import/export entry. New assignment4 files provide a small-model Transformers adapter, deterministic sampling, checkpoints, a driver and analysis. Generation/extraction and EvalPlus testing are reused from upstream commit ce22fe2f80c8f36da85cc8a0cca49c2bcc8b1cb9. The new driver tests before the reverse step, following the course package; current upstream's batch driver instead runs both tasks before testing. Both fresh conditions use the new driver and the package's exact generation/prose templates, which differ from current upstream defaults. Infrastructure errors abort rather than count as model failures. A final reverse output is logged after cycle 10 but is not used.

## 3. Experimental setup and scoring
Four runs evaluate the two models under standard and pseudocode loops on identical 100-task subsets. Task IDs are sorted by SHA256 of "evaloop-a4-2026:" plus ID; the first 100 are selected without consulting outcomes. Each configuration uses one greedy run (do_sample=False), a 1,024-new-token limit per call, float16 weights, seed 2026 and ten-cycle cap. The native user-only chat template is used. All runs used a Tesla T4, Python 3.12, PyTorch 2.6.0+cu124, Transformers 4.48.3 and EvalPlus 0.3.1. Full model revisions, dataset hash, task IDs and source fingerprint are recorded in JSON. Exact cycle-1 outputs match across conditions for both models.

Success is pass@cycle-1. For sustained count s_t and N=100, reported ASL_pow = sum(s_t squared)/(10N), matching the package's aslScore. It is not the ordinary mean or upstream's semantic-weighted ASL. No judge is used in scoring. Tests ran through EvalPlus subprocess guards/timeouts in a disposable Colab VM. Recorded evaluation time totals 3.47 hours, excluding setup/downloads, with 3,424 model calls and 933,701 input-plus-output tokens. API charges were $0; any Colab subscription/compute charge is not known from the records.

<!-- PAGEBREAK -->
## 4. Results
| Model | Loop | N | Success | ASL_pow | Pass@10 |
|---|---|---:|---:|---:|---:|
| Qwen 0.5B | Standard | 100 | 45% | 3.652 | 36% |
| Qwen 0.5B | Pseudocode | 100 | 45% | 3.123 | 31% |
| Qwen 1.5B | Standard | 100 | 51% | 5.001 | 50% |
| Qwen 1.5B | Pseudocode | 100 | 51% | 3.717 | 37% |

Pseudocode lowers ASL by 0.529 (14.5%) for 0.5B and 1.284 (25.7%) for 1.5B. The ordering remains 1.5B > 0.5B, but the ASL gap shrinks from 1.349 to 0.594. Among initially correct tasks, ten-cycle retention falls from 80.0% to 68.9% for 0.5B and from 98.0% to 72.5% for 1.5B. Thus the standard loop's near-perfect retention for 1.5B does not generalize to this representation. At task level, pseudocode sustains more/fewer/equal cycles on 8/14/78 tasks for 0.5B and 1/14/85 for 1.5B. Equal outcomes include many first-cycle failures, which cannot be attributed to the reverse task.

The run records passed independent structural checks: identical manifests, matching source fingerprints, correct stop rules, base-and-Plus conjunction, immutable assertion retention, exact next-input propagation and consistent call counts. Recomputed CSV and paired metrics exactly match the Colab exports. These checks establish record consistency, not independent reexecution of generated programs or proof that the benchmark tests are complete.

## 5. Failure analysis — provisional until student review
The attached review packet examines the first ten post-first-cycle extension failures per model in the fixed task order. These are 20 model/task loops, with some task IDs shared across models. Each case includes the original task, last passing code, intervening pseudocode, first failing code, and paired standard-loop evidence. The assessments below were drafted by the coding assistant through textual inspection, not by an evaluation-time classifier; they require student verification before being described as completed manual review.

| Primary suggested category | Cases | Evidence pattern |
|---|---:|---|
| Algorithm drift | 9 | Missing operations, inverted branches or changed algorithm |
| Invented restrictions | 5 | New input checks or unjustified boundary rules |
| Output-structure drift | 2 | Correct return value wrapped in the wrong container |
| Interface drift | 2 | Expected function name changed despite retained assertion |
| Implementation error | 2 | Formula retained but required import omitted |

Categories are exclusive primary labels; several cases have secondary defects. Six reviewed failures still pass base tests and fail Plus. Nineteen matched standard loops survive ten cycles; the remaining one survives six. These selected cases characterize mechanisms, not unbiased category frequencies across all failures. Exact failing Plus inputs and exception traces are not logged: claimed defects are static observations, and individual triggering tests are not asserted.

A representative extension failure is Qwen 1.5B, Mbpp/273. Cycle 1 uses zip-based tuple subtraction and passes. Its pseudocode invents ValueError for empty or unequal-length tuples. Cycle 2 implements those checks, passes base and fails Plus; the matched standard run survives ten cycles. The original zip behavior permits those inputs. This illustrates a new restriction introduced by a prompt that explicitly asked the model not to add requirements.

<!-- PAGEBREAK -->
## 6. Contrasts and interpretation
On Qwen 1.5B, Mbpp/12, passing code is sorted(matrix, key=sum). Pseudocode claims sorted rows but instructs the generator to append scalar row sums without sorting. Cycle 2 follows those steps and fails both suites; standard survives ten cycles. The evidence suggests that detailed procedural steps can transmit a concrete wrong algorithm even while an OUTPUT heading retains the intended behavior. This is more specific than simply saying the summary forgot a requirement.

On Qwen 0.5B, Mbpp/422, cycle-2 pseudocode changes find_Average_Of_Cube to find_average_of_cube; cycle-3 code follows the new spelling despite the retained assertion. Mbpp/273 similarly changes substract_elements to subtract_elements and also changes tuple output to list. Therefore the inherited assertion mechanism reduces interface drift but does not eliminate it. On 0.5B Mbpp/276 and Mbpp/82, regenerated code references math.pi without importing math; those failures should not be counted solely as semantic-description loss.

The effect is not uniformly negative. On Qwen 1.5B, Mbpp/730, the standard loop fails at cycle 2 by omitting the first element when removing consecutive duplicates. The pseudocode loop preserves the explicit i == 0 or changed-from-previous condition and survives all ten cycles. Its prose OUTPUT claims remain imprecise, so survival shows functional success under these tests rather than perfect descriptions. This is a counterexample to a blanket claim that pseudocode always hurts. On Mbpp/132, both models and both conditions survive all ten cycles when concatenating tuple elements, although adding str conversion broadens behavior beyond the original join implementation.

The inspected baseline failure for Qwen 0.5B, Mbpp/290, occurs at cycle 7: regenerated code reuses the loop variable as the selected maximum list, returning the last list. The summary still describes the intended maximum. In the extension, cycle-1 pseudocode instead nests the selected list and cycle 2 copies it. This paired case distinguishes incorrect reconstruction of a reasonable description from an explicit new defect introduced in the reverse representation. Neither loop is immune to either mechanism.

## 7. Limitations and implications
Only 100 public tasks and two sizes of one model family were evaluated; contamination and task selection limit generalization. One greedy run per condition does not estimate run-to-run variance, and two models cannot establish a general size effect. GPU kernels, precision and output limits are part of the protocol. The pseudocode prompt also changes length and instructional detail, so the experiment does not isolate structure from verbosity. Pseudocode runs consume 128,537 and 173,540 more total tokens than their 0.5B and 1.5B baselines respectively, despite fewer model calls. The reverse prompt's emphasis on exceptional behavior may encourage invented validation, but that mechanism is an interpretation rather than a separately tested ablation.

Finite tests can accept changed semantics. For example, Mbpp/140's displayed assertion uses tuples while its passing first implementation concatenates using a list seed; passing the recorded tests does not prove it satisfies every displayed example. Retained assertions can anchor or bias the solution, yet some generated programs ignore them. Sampled failure categories are subjective and biased toward early post-first-cycle failures; no independent reviewer agreement is available. EvalPlus guards are not a hardened security boundary. Within these limits, the experiment shows representation-sensitive self-consistency: high robustness under ordinary summaries does not ensure the same robustness under explicit procedural descriptions.

## References and assistance
EvaLoop: arxiv.org/abs/2505.12185; code: github.com/evalooop/EvaLooop. EvalPlus: github.com/evalplus/evalplus. Models: huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct and Qwen2.5-Coder-1.5B-Instruct. The course data package supplies the standard templates and unweighted metric convention; its records are not used as these models' baselines or redistributed. Coding-assistant help covered implementation, draft analysis and writing; the exact assistant backend/version and per-example cost are not exposed here. No assistant decisions enter test outcomes or ASL. Student review and disclosure must follow course policy.
