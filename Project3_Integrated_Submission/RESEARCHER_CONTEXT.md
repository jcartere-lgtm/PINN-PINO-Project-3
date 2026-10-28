## Researcher interview and written-brief status

**Interview:** October 16, 2026, with Dr. Jimmie Adriazola.
**Source:** the team's account of that interview, supplied for this report.
The statements below are paraphrases of the team's notes, not a verbatim
transcript or approval of this written brief.

| Topic | What the team reports Dr. Adriazola stated | What remains unavailable |
|---|---|---|
| Current numerical approach | His published work uses traditional numerical methods, including a pseudo-spectral Fourier split-step method with Strang splitting for the Gross–Pitaevskii equation. He mentioned exponential time differencing as an alternative. One-dimensional solves are relatively inexpensive. | Exact runtime, hardware specification, and a benchmark for his current research solver were not provided. |
| Repeated control queries | PDE-constrained optimization repeatedly evaluates the PDE while adjusting controls, for example through gradient descent or Newton-type methods. He proposed investigating whether an operator could learn the control-parameterized solution map and accelerate repeated optimization queries. | Typical PDE-solve count, iteration count, and number of control queries were not specified. |
| Useful physical outcome | He recommended squeezing and splitting a wave packet, emphasizing approximately stationary packets after optimization. | No numerical accuracy threshold was specified. Achieved fidelity, control regularization, numerical solution error, and post-control stability must be reported without claiming researcher-approved accuracy. |
| Problem selection and follow-up | He personally proposed the GPE control problem, directed the team to Equations (2)–(4) of his paper, recommended a conventional-versus-operator comparison, offered help, and requested that the team share its results. | He has not yet reviewed or approved a written problem brief. Memo delivery, any response, and permission to publish interview material have not been reported. |

**Scope chosen by the team:** this minimal case studies squeezing. Splitting
was discussed in the interview but is not implemented in the current experiment.
The post-control hold measures changes in density shape and width; a stationary
GPE state may still accumulate an overall complex phase. A finite hold and a
particular achieved error do not establish a researcher-approved usefulness level.

**Evidence boundaries:** interview statements describe the researcher's problem
and proposed question. Every runtime, evaluation count, fidelity, loss, error,
and break-even estimate reported for this project comes from the team's saved
local experiments, not from a benchmark supplied by Dr. Adriazola. No useful
neural-operator speedup has been established. The contact method, researcher
reservation record, and any earlier outreach or rehearsal activity were not
provided; the October 16 interview date is recorded as supplied.
