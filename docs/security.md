# Security notes

## Supply chain

Every push runs `pip-audit` over the fully exported, locked dependency set — all extras
included, not just what the test job installs. Dependabot opens grouped pull requests weekly,
and `gitleaks` runs as a pre-commit hook so a key cannot reach a public repository by
accident.

## Accepted risks

An audit that is silenced without a reason is worse than no audit. Every ignored advisory is
listed here with the argument for ignoring it and the condition that ends it.

### PYSEC-2026-3552 / CVE-2026-69247 — cryptography 49.x

* **What it is:** PKCS#7 `EnvelopedData` decryption exposes a Bleichenbacher oracle through
  distinguishable errors and timing.
* **Why it is accepted:** this project never decrypts PKCS#7 envelopes. `cryptography` arrives
  transitively via `mlflow` and `google-auth`, which use it for TLS and token verification.
  The vulnerable code path is not reachable from anything here.
* **Why it is not simply fixed:** the fix is `cryptography>=50.0.0`, and `mlflow` 3.15.1 —
  the newest release, which fixes all of its own outstanding advisories — pins
  `cryptography<50`. Forcing the upgrade makes the dependency set unsatisfiable. Choosing the
  other way round would mean running an mlflow with roughly twenty open advisories to close
  one unreachable advisory in a transitive library.
* **Revisit when:** mlflow relaxes its upper bound. Then drop the `--ignore-vuln` flag from
  the workflow and delete this entry.

## Application security

Implemented as the relevant iterations land, and tested rather than asserted:

| Concern | Approach | Iteration |
|---|---|---|
| Indirect prompt injection | Retrieved text is fenced as data, never as instructions; a poisoned-chunk fixture must not change behaviour | 3 |
| Ungrounded citations | A citation pointing at a block that was never retrieved raises `UngroundedCitationError` | 3 |
| Secrets | Environment only, `.env` gitignored, `.env.example` documents the shape | 0 |
| Authentication | API-key header on every non-health route; managed identity on the Azure path | 5, 8 |
| Denial of service | Request size limits, a step cap on the agent, rate limiting | 5, 6 |
| Container hardening | Multi-stage build, slim base, non-root user | 5 |
| Tool access control | Explicit tool allowlist and an approval mode for sensitive tools | 6 |
