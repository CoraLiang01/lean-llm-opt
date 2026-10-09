Let $T$ be the set of 48 half-hour intervals indexed in order as $t=1,\ldots,48$, each with required minimum waitstaff $r_t$ from 44.csv (column "Requirement", table_id: file_0_view_0, row $t-1$). Let $x_s$ be the number of waitstaff starting work at interval $s$ ($s=1,\ldots,48$). Each waitstaff works 16 consecutive intervals (8 hours).

Sets:
$T = \{1,2,\ldots,48\}$

Parameters:
$r_t$ = required waitstaff in interval $t$ (from 44.csv, "Requirement", table_id: file_0_view_0, row $t-1$)

Decision variables:
$x_s \in \mathbb{Z}_{\geq 0}$, number of waitstaff starting at interval $s$ ($s \in T$)

Objective:
Minimize total number of waitstaff:
$\min \sum_{s=1}^{48} x_s$

Constraints:
For each interval $t \in T$,
$\sum_{s=1}^{48} a_{t,s} x_s \geq r_t$

where
$a_{t,s} = 1$ if a waitstaff starting at $s$ is working during interval $t$, $0$ otherwise.

Specifically,
$a_{t,s} = 1$ if $t \in \{s, s+1, \ldots, s+15\}$ modulo 48 (i.e., intervals wrap around midnight), else $0$.

Variable domains:
$x_s \geq 0$ and integer, for all $s \in T$

Data Mapping:
- $T$: all 48 intervals from 44.csv, column "Time", table_id: file_0_view_0, rows 0–47
- $r_t$: 44.csv, column "Requirement", table_id: file_0_view_0, row $t-1$
- $x_s$: decision variable, number of waitstaff starting at interval $s$
- $a_{t,s}$: defined as above, using interval indices modulo 48

Summary:
Minimize $\sum_{s=1}^{48} x_s$

subject to, for all $t=1,\ldots,48$:
$\sum_{s: t \in \{s, s+1, \ldots, s+15\} \bmod 48} x_s \geq r_t$

$x_s \in \mathbb{Z}_{\geq 0}$ for all $s=1,\ldots,48$