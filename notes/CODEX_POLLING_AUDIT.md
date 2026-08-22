# Codex polling audit (2026-08-22)

Scope: the single local session available at audit time. This is evidence of the mechanism, not a statistically reliable estimate of dollar savings.

- Custom tool calls observed: 34 (18 other exec calls, 16 `write_stdin`).
- Empty `write_stdin` polls: 12.
- Empty/non-interactive polls with `yield_time_ms=1000`: at least 8; ten total 1-second `write_stdin` calls include interactive writes.
- Session cumulative logical input at audit time: 1,444,078 tokens.
- Cached input: 1,338,112 tokens (92.66% of logical input).
- Non-cached input: 105,966 tokens.
- Output: 5,133 tokens; reasoning output: 877 tokens.

The dependency install alone caused several consecutive 1-second empty polls, including polls returning no output. A single 180-300 second poll would have returned early on process completion and eliminated those wakeups. Exact billed savings cannot be inferred from logical token totals alone because cached input is priced differently, and one session cannot validate a claimed 25% cross-project saving.

Mitigation installed at `/root/.codex/AGENTS.md` using the supplied long-poll rules. Official documentation says global instructions belong under the Codex home directory and are loaded once per run, with nearer project instructions taking precedence.
