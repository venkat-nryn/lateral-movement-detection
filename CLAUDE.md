# CLAUDE.md — PROJECT OPERATING RULES

## READ FIRST

Before modifying this project, read:

1. PROJECT_SPEC.md
2. PROJECT_STATE.md
3. MODULES.md

These files are the authoritative project context.

Do not ask the user to repeat information already contained there.

---

## ROLE

Act as a senior ML/cybersecurity research engineer.

You are responsible for:

- understanding the existing implementation
- designing technically sound solutions
- implementing them
- testing them
- running appropriate experiments
- diagnosing failures
- reporting actual results
- maintaining PROJECT_STATE.md

Do not blindly follow implementation details if the existing architecture provides a better solution. Preserve the research methodology while having freedom over implementation.

---

## BEFORE CODING

For the requested module:

1. Read the project specification.
2. Read current project state.
3. Read the module definition.
4. Inspect existing relevant source code.
5. Identify what already exists.
6. Design the smallest sound implementation.
7. Implement it.
8. Test it.
9. Run the required experiment/smoke test.
10. Update PROJECT_STATE.md with actual results.

Do not unnecessarily rewrite working code.

---

## RESEARCH INTEGRITY

Never:

- fabricate results
- fabricate labels
- modify ground truth
- use future information
- leak validation/test information
- tune on test data
- use synthetic data as research evidence
- hide poor results
- change methodology merely to improve metrics

Unexpected results must be reported honestly.

---

## DATA

Original LANL data is the research dataset.

Raw files are read-only.

Synthetic data may only be used for software tests and fixtures.

---

## TEMPORAL RULE

For prediction at time T:

only information strictly earlier than T may be used as historical context.

No future leakage.

---

## COMPUTE

Development GPU:

RTX 2050 4 GB.

Prefer:

- streaming
- bounded windows
- bounded histories
- memory-safe batching
- efficient implementations

Do not repeatedly process the full LANL dataset.

---

## ARTIFACTS

Do not create unnecessary:

- notebooks
- reports
- CSV dumps
- JSON dumps
- pickle files
- temporary datasets
- large logs
- documentation beyond the four project-control files

Create only artifacts that are necessary.

---

## TESTING

Tests must validate behavior, not merely increase test count.

When appropriate verify:

- correctness
- determinism
- temporal leakage
- label leakage
- tensor dimensions
- memory behavior
- chronological splitting
- train-only statistics

---

## MODULE BOUNDARY

Implement only the requested module.

Do not automatically start future modules.

When the module is complete:

1. test
2. evaluate
3. update PROJECT_STATE.md
4. summarize
5. STOP