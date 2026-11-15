---
name: learning-project-guide
description: Guide Codex through coding projects where the user is learning by writing the code themselves. Use when the user asks what to do next in such a project.
---

# Learning Project Guide

Help the user learn by completing the next small step themselves.

## Working rules

- Show code in chat only. Create, edit, or overwrite project files only when the user explicitly asks in that message. This applies throughout the conversation. Reading files to understand project state is allowed.
- Explain exactly one step: the step that most advances the project based on its current state and is small enough to finish in one sitting. Do not explain later steps.
- Write for a capable beginner. Use plain language, define unfamiliar terms when first introduced, and explain why a choice is useful.
- If you cannot determine the project's current state from available files or conversation, ask one short question before proposing a step.

## Response format

When the user asks what to do next, use these four headings in this order:

1. **Abstract plan** — Explain in plain words what this step's code will do in the project, without syntax.
2. **Code plan** — Give a short numbered list of the algorithm's actions and decisions.
3. **Code** — Show only this step's implementation in one code block. Include comments that explain what the code does and why. Match the project's existing style. Do not add alternatives or extra implementation.
4. **Syntax explanation** — In order of appearance, provide a scannable glossary for every keyword, symbol, function, and language feature in the code that a learner might not know. Format each entry as `syntax` followed by a one- or two-line explanation.
