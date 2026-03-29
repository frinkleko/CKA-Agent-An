**Table R1.** Comparison with prior tree-search jailbreak methods on StrongREJECT (target: GPT-oss-120B, judge: Gemini-2.5-Flash).

| Method     | Full Success (FS)↑   | Partial Success (PS)↑   | Vacuous (V)↓    | Refusal (R)↓    |
|------------|-------|-------|-------|-------|
| TAP        | 0.095 | 0.032 | 0.016 | 0.857 |
| DAMON + Same Synthesizer      | 0.654 | 0.111 | 0.068 | 0.167 |
| CKA-Agent  | 0.982 | 0.012 | 0.006 | 0.000 |


**Table R2.** Comparison with guardrail-evasion baselines on StrongREJECT (target: GPT-oss-120B, judge: Gemini-2.5-Flash).

| Method      | FS↑   | PS↑   | V↓    | R↓    |
|-------------|-------|-------|-------|-------|
| FlipAttack  | 0.012 | 0.049 | 0.049 | 0.889 |
| JAM         | 0.019 | 0.010 | 0.031 | 0.941 |
| JAIL-CON    | 0.278 | 0.142 | 0.056 | 0.525 |
| CKA-Agent   | 0.982 | 0.012 | 0.006 | 0.000 |

**Table R3.** Context-aware defense evaluation. FS↑ is reported. (Judge: Gemini-2.5-Flash)

| Dataset      | Defense Setting                        | Gemini-2.5-Flash | Gemini-2.5-Pro | GPT-oss | Claude-Haiku-4.5 |
|--------------|----------------------------------------|-----------------:|---------------:|--------:|-----------------:|
| StrongREJECT | CKA-Agent-Branch                      | 0.969            | 0.937          | 0.846   | 0.956            |
| StrongREJECT | CKA-Agent-Primed (Intention Analysis) | 0.883            | 0.852          | 0.821   | 0.864            |
| HarmBench    | CKA-Agent-Branch                      | 0.921            | 0.960          | 0.786   | 0.889            |
| HarmBench    | CKA-Agent-Primed (Intention Analysis) | 0.857            | 0.841          | 0.730   | 0.786            |

**Table R4.** Ablation on maximum search depth (StrongREJECT, target: GPT-oss-120B, judge: Gemini-2.5-Flash).

| Max Depth | FS↑   | PS↑   | V↓    | R↓    |
|-----------|-------|-------|-------|-------|
| 3         | 0.939 | 0.049 | 0.006 | 0.006 |
| 5         | 0.982 | 0.012 | 0.006 | 0.000 |
| 7         | 0.988 | 0.012 | 0.000 | 0.000 |

**Table R5.** Category-level performance on StrongREJECT (target: GPT-oss-120B, judge: Gemini-2.5-Flash).

| Category                          | ASR↑  |
|-----------------------------------|-------|
| Illegal goods & services          | 1.00  |
| Non-violent crimes                | 1.00  |
| Violence                          | 1.00  |
| Disinformation & deception        | 0.98  |
| Sexual content                    | 0.96  |
| Hate, harassment & discrimination | 0.92  |

**Table R6.** Full StrongREJECT benchmark evaluation (target: GPT-oss-120B, judge: Gemini-2.5-Flash).

| Setting                 | FS↑   | PS↑   | V↓    | R↓    |
|-------------------------|-------|-------|-------|-------|
| Complete StrongREJECT   | 0.952 | 0.026 | 0.003 | 0.019 |

**Table R7.** Evaluation on newer frontier models (judge: Gemini-2.5-Flash).

| Dataset      | Model          | FS↑   | PS↑   | V↓    | R↓    |
|--------------|----------------|-------|-------|-------|-------|
| StrongREJECT | GPT-5.2        | 0.932 | 0.056 | 0.006 | 0.006 |
| StrongREJECT | Gemini-3.0-Pro | 0.951 | 0.037 | 0.006 | 0.006 |
| HarmBench    | GPT-5.2        | 0.889 | 0.079 | 0.024 | 0.008 |
| HarmBench    | Gemini-3.0-Pro | 0.881 | 0.087 | 0.000 | 0.032 |
