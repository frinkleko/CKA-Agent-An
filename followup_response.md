**Table R2.** Comparison of guardrail-evasion baselines and CKA-Agent under the same augmented defense settings (Circuit Breakers, LLM Guard) as Figure 4 on Gemini-2.5-Flash.

| Defense          | Method     | FS↑    | PS↑    | V↓     | R↓     |
|------------------|------------|--------|--------|--------|--------|
| Circuit Breakers | FlipAttack | 0.0000 | 0.0079 | 0.0000 | 0.9921 |
| Circuit Breakers | JAIL-CON   | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| Circuit Breakers | JAM        | 0.2188 | 0.1016 | 0.3125 | 0.3516 |
| Circuit Breakers | CKA-Agent  | **0.8730** | 0.0952 | 0.0317 | 0.0000 |
|------------------|------------|--------|--------|--------|--------|
| LLM Guard        | FlipAttack | 0.0159 | 0.0000 | 0.0000 | 0.9841 |
| LLM Guard        | JAIL-CON   | 0.0714 | 0.0556 | 0.0238 | 0.8492 |
| LLM Guard        | JAM        | 0.2302 | 0.1032 | 0.2460 | 0.4206 |
| LLM Guard        | CKA-Agent  | **0.9840** | 0.0080 | 0.0080 | 0.0000 |
