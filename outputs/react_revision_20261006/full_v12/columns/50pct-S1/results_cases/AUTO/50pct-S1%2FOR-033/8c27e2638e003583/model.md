Let $T$ be the set of 48 half-hour intervals indexed in order as $t=1,\ldots,48$, each with required minimum staff $r_t$ from the Requirement column of table_id file_0_view_0. Let $x_s$ be the number of waitstaff starting work at interval $s$ ($s=1,\ldots,48$). Each staff works 16 consecutive intervals (8 hours).

Minimize
$$
\sum_{s=1}^{48} x_s
$$

Subject to, for all $t=1,\ldots,48$:
$$
\sum_{s: (t-s) \bmod 48 \in \{0,1,\ldots,15\}} x_s \geq r_t
$$

$$
x_s \geq 0,\quad x_s \in \mathbb{Z} \quad \forall s=1,\ldots,48
$$

Data Mapping:
- $T$ (intervals): file_0_view_0, column "Time", rows 0–47
- $r_t$: file_0_view_0, column "Requirement", rows 0–47
- $x_s$: decision variable, number of staff starting at interval $s$