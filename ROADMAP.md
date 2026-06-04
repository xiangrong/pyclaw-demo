# PyClaw Development Roadmap

Based on benchmarking against **OpenClaw** and **Hermes Agent (Nous Research)**, this roadmap outlines the evolution of PyClaw into a persistent, self-improving AI Operating System.

## Phase 1: Foundational Persistence & Security (Short-term)
*Goal: Move from session-based tasks to a long-lived identity with secure execution.*

- [ ] **Sandboxed Execution**: Refactor `TerminalTool` to support optional Docker containers or SSH remotes to prevent local system compromise.
- [ ] **Durable Knowledge Files**:
    - Implement `MEMORY.md`: A curated file where the agent stores long-term facts, standing decisions, and completed project summaries.
    - Implement `USER.md`: Stores static context about the user (preferences, bio, recurring needs).
- [ ] **Human-in-the-loop (HITL)**: Add an approval mechanism for "High Risk" tools (e.g., `rm -rf`, `terminal` commands, large file deletions). The agent must wait for a user confirmation via the messaging app before proceeding.

## Phase 2: Self-Improvement & Learning (Medium-term)
*Goal: Enable the agent to expand its own capabilities through experience.*

- [x] **Closed Learning Loop (Autonomous Skills)**: 
    - Enable the agent to identify a "missing capability" and write its own `SKILL.md` (Markdown instructions) or `.py` tool script.
    - Add a `save_as_skill` tool that persists successful complex procedures for future use.
- [ ] **Semantic Memory**:
    - Integrate a lightweight Vector DB (e.g., LanceDB or ChromaDB) to enable full-text and semantic search across *all* past sessions and uploaded documents.
- [ ] **Dynamic Skill Indexing**: Optimize the prompt by only injecting the full documentation of a skill *after* the agent decides to use it (Progressive Disclosure).

## Phase 3: Proactive Intelligence & Ecosystem (Long-term)
*Goal: Transition from a reactive assistant to a proactive partner.*

- [x] **Natural Language Heartbeat**: Expand the Cron system to allow "Natural Language Automations" (e.g., "Check my email every morning and summarize the ones from my boss").
- [ ] **Multi-Agent Swarms**:
    - Implement a more robust `SubAgentTool` where sub-agents can have their own isolated workspaces, specialized `SOUL.md` files, and parallel execution paths.
- [ ] **Extended Gateway Connectivity**:
    - Add connectors for Discord, Slack, and WhatsApp (via Matrix or official APIs) to achieve the "Universal Reach" of Hermes.
- [ ] **Model-Agnostic Reasoning**: Optimize the agent loop for small local models (e.g., Llama 3.1 8B) using specialized XML tags (`<thought>`, `<action>`) similar to Hermes 3.

---
*Last Updated: 2026-06-02*
