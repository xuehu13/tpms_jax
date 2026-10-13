# r42: saved-state initial energy accounting

Read-only reassessment of r30/r34/r40 smooth and binary equilibria for diverse_04. No new equilibrium solve, Abaqus job, production change, path or design AD.

- [Frozen protocol](protocol.json), [source protection hashes](frozen_before.json), [analysis recipe](analyze.py), [result](result.json).
- Original 27: fixed-state net weight change −0.8643%, re-equilibration releases 3.1440%, net −4.0083%.
- Same selected12/rest27 rule: −0.6217%, releases 1.4988%, net −2.1205%.
- Denominator: smooth equilibrium energy under each SAME rule, not shell K. Outside r33 selection not densely checked. Exact variational accounting does not uniquely attribute physical shell error or certify sharp-interface convergence.
- 36.73 s post-processing; saved energies and compatible-difference identity passed frozen checks. Historical failures remain.

![Energy accounting](energy_balance.png)

Interpretation is maintained only in [current mechanism review](../../docs/MECHANISM_ANALYSIS.md); execution only in [unique plan](../../docs/RESEARCH_PLAN.md).
