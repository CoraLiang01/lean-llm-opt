Let $I = \{A1, A2, \ldots, A80\}$ be the set of products.

Parameters (for each $i \in I$):

- $D_i$: Maximum demand (100 kg units)
- $p_i$: Selling price ($/100 kg$)
- $c_i$: Production cost ($/100 kg$)
- $q_i$: Production quota per day (max 100 kg units/day)
- $f_i$: Fixed activation cost ($)
- $b_i$: Minimum batch size (100 kg units)
- $T = 22$: Number of production days

Decision variables (for each $i \in I$):

- $x_i \in \mathbb{Z}_+$: Number of 100 kg units of product $i$ produced in the month
- $y_i \in \{0,1\}$: 1 if product $i$'s production line is activated, 0 otherwise

Model:

Maximize total profit:
$$
\max \sum_{i \in I} \left[ p_i x_i - c_i x_i - f_i y_i \right]
$$

Subject to, for all $i \in I$:
\begin{align*}
& x_i \leq D_i \\
& x_i \leq q_i \cdot T \\
& x_i \geq b_i y_i \\
& x_i \leq D_i y_i \\
& x_i \in \mathbb{Z}_+ \\
& y_i \in \{0,1\}
\end{align*}

Where:
- $x_i$ is the integer number of 100 kg units produced of product $i$.
- $y_i$ is 1 if product $i$ is produced, 0 otherwise.
- If $y_i = 0$, then $x_i = 0$.
- If $y_i = 1$, then $x_i \geq b_i$ and $x_i \leq \min(D_i, q_i T)$.

---

#### Retrieved Information

- Maximum Demand (100 kg units):

| Product | Value | ... | Product | Value |
|---------|-------|-----|---------|-------|
| A1 | 5732 | ... | A80 | 4466 |

- Selling Price ($/100 kg):

| Product | Value | ... | Product | Value |
|---------|-------|-----|---------|-------|
| A1 | 111 | ... | A80 | 106 |

- Production Cost ($/100 kg):

| Product | Value | ... | Product | Value |
|---------|-------|-----|---------|-------|
| A1 | 71.38 | ... | A80 | 67.19 |

- Production Quota (max per day, 100 kg units):

| Product | Value | ... | Product | Value |
|---------|-------|-----|---------|-------|
| A1 | 586 | ... | A80 | 556 |

- Activation Cost ($):

| Product | Value | ... | Product | Value |
|---------|-------|-----|---------|-------|
| A1 | 178539 | ... | A80 | 146842 |

- Minimum Batch Size (100 kg units):

| Product | Value | ... | Product | Value |
|---------|-------|-----|---------|-------|
| A1 | 18 | ... | A80 | 18 |

(All values for A1 through A80 are as retrieved and in source order.)

---

Explicitly, for each $i \in I$ (A1 to A80):

- $D_i$ = value from "Maximum Demand (100 kg units)" row
- $p_i$ = value from "Selling Price ($/100 kg)" row
- $c_i$ = value from "Production Cost ($/100 kg)" row
- $q_i$ = value from "Production Quota (max per day)" row
- $f_i$ = value from "Activation Cost ($)" row
- $b_i$ = value from "Minimum Batch Size (100 kg units)" row

All coefficients and identifiers are preserved in source order.