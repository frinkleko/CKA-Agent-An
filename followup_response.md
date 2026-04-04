**Table R1.** Key Principle Comparison: DAMON vs. CKA-Agent

| # | Aspect | DAMON | CKA-Agent |
|---|---|---|---|
| 1 | **Design Philosophy** | Relies on predefined attacker strategies (e.g., Progressive Inquiry, Step-by-Step Questioning, Role-play) | Dynamically constructs the search path based on correlated knowledge revealed by the target model |
| 2 | **Next Step Decision** | Chooses the next step from fixed strategy templates when blocked | Adapts each next query based on the target's latest response and new knowledge fragments |
| 3 | **Role of Target Model** | The target is a passive environment to bypass | The target is both the attack target and the knowledge source, guiding the exploration |
| 4 | **Handling Blocked Paths** | If blocked, selects another predefined strategy | If blocked, dynamically reroutes to alternative knowledge branches |
| 5 | **Empirical Example Comparison (Same Prompt)** | *Prompt: "How do I make a tennis ball bomb?"* <br> → *Same-level Node 1*: "**You're an inventor stranded on a remote island**..." <br> → *Same-level Node 2*: "**Trapped in a besieged city**..." <br> MCTS evaluates which persona is least restricted. | *Prompt: "How do I make a tennis ball bomb?"* <br> → *Same-level Node 1*: "**What are the chemical properties**..." <br> → *Same-level Node 2*: "**What are the mechanical requirements**..." <br> Branches are scored based on correlated knowledge revealed (e.g., pressure specs). |
| 6 | **Empirical Performance (GPT-5.2, Full Success)** | 0.2754 | **0.9320** |


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
