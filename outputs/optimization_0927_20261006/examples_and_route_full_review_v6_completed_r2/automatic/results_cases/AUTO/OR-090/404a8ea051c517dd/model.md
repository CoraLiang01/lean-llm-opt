Let $x_i$ be the number of batches of product $i$ to produce or purchase, for $i \in \{\text{P1}, \text{P2}, \ldots, \text{P100}\}$. Each batch is 10 units. All $x_i$ are nonnegative integers.

**Parameters (from data):**

- For each product $i$:
    - $\text{profit\_per\_unit}_i$: profit per unit
    - $\text{r1\_per\_unit}_i$: units of R1 consumed per unit
    - $\text{r2\_per\_unit}_i$: units of R2 consumed per unit
    - $\text{r3\_per\_unit}_i$: units of R3 consumed per unit
    - $\text{upper\_demand\_units}_i$: maximum market demand (units)
    - $\text{batch\_size\_units}_i = 10$
- Resource capacities:
    - $C_{R1} = 27380.54$
    - $C_{R2} = 22245.11$
    - $C_{R3} = 15147.73$

**Decision variables:**

- $x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in \{\text{P1}, \ldots, \text{P100}\}$

---

**Objective:**

\[
\max \sum_{i=1}^{100} 10 \cdot x_i \cdot \text{profit\_per\_unit}_i
\]

---

**Constraints:**

For each resource $j \in \{\text{R1}, \text{R2}, \text{R3}\}$:

\[
\sum_{i=1}^{100} 10 \cdot x_i \cdot \text{r}j\_\text{per\_unit}_i \leq C_{Rj}
\]

That is,

\[
\sum_{i=1}^{100} 10 \cdot x_i \cdot \text{r1\_per\_unit}_i \leq 27380.54
\]
\[
\sum_{i=1}^{100} 10 \cdot x_i \cdot \text{r2\_per\_unit}_i \leq 22245.11
\]
\[
\sum_{i=1}^{100} 10 \cdot x_i \cdot \text{r3\_per\_unit}_i \leq 15147.73
\]

For each product $i$:

\[
10 \cdot x_i \leq \text{upper\_demand\_units}_i
\]

\[
x_i \in \mathbb{Z}_{\geq 0}
\]

---

**Full Data Table (source order):**

| product | profit_per_unit | r1_per_unit | r2_per_unit | r3_per_unit | upper_demand_units | batch_size_units |
|---------|----------------|-------------|-------------|-------------|--------------------|------------------|
| P1      | 6.7            | 2.37        | 0.61        | 2.03        | 317                | 10               |
| P2      | 10.96          | 4.79        | 2.73        | 0.53        | 106                | 10               |
| P3      | 9.4            | 3.87        | 1.6         | 0.74        | 386                | 10               |
| P4      | 11.13          | 3.31        | 2.28        | 2.73        | 441                | 10               |
| P5      | 10.29          | 1.46        | 3.68        | 1.94        | 63                 | 10               |
| ...     | ...            | ...         | ...         | ...         | ...                | ...              |
| P100    | 8.84           | 1.25        | 3.23        | 0.53        | 576                | 10               |

Resource capacities:

| resource | capacity   |
|----------|-----------|
| R1       | 27380.54  |
| R2       | 22245.11  |
| R3       | 15147.73  |

---

**Complete Model:**

\[
\begin{align*}
\max\ & \sum_{i=1}^{100} 10 \cdot x_i \cdot \text{profit\_per\_unit}_i \\
\text{s.t.} \quad
& \sum_{i=1}^{100} 10 \cdot x_i \cdot \text{r1\_per\_unit}_i \leq 27380.54 \\
& \sum_{i=1}^{100} 10 \cdot x_i \cdot \text{r2\_per\_unit}_i \leq 22245.11 \\
& \sum_{i=1}^{100} 10 \cdot x_i \cdot \text{r3\_per\_unit}_i \leq 15147.73 \\
& 10 \cdot x_i \leq \text{upper\_demand\_units}_i \quad \forall i \in \{\text{P1},\ldots,\text{P100}\} \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{\text{P1},\ldots,\text{P100}\}
\end{align*}
\]

All coefficients and identifiers are as in the retrieved data, in source order.