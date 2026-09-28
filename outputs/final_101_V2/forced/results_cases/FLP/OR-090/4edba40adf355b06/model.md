##### Decision Variables

$x_i \in \mathbb{Z}_+, \quad i=1,\ldots,100$: number of batches of product $i$ to produce (each batch = 10 units).

##### Parameters

Let $I = \{P1, P2, \ldots, P100\}$ (product set).

For each product $i \in I$:
- $\text{profit\_per\_unit}_i$: profit per produced unit (from CSV)
- $\text{r1\_per\_unit}_i$: resource R1 consumed per unit
- $\text{r2\_per\_unit}_i$: resource R2 consumed per unit
- $\text{r3\_per\_unit}_i$: resource R3 consumed per unit
- $\text{upper\_demand\_units}_i$: maximum market demand (units)
- $\text{batch\_size\_units} = 10$

Resource capacities:
- $C_1 = 27380.54$ (R1)
- $C_2 = 22245.11$ (R2)
- $C_3 = 15147.73$ (R3)

##### Objective Function

\[
\max \sum_{i \in I} 10 \cdot x_i \cdot \text{profit\_per\_unit}_i
\]

##### Constraints

1. **Resource constraints** (for $k=1,2,3$):

\[
\sum_{i \in I} 10 \cdot x_i \cdot \text{r}k\_\text{per\_unit}_i \leq C_k
\]
That is,
\[
\sum_{i \in I} 10 \cdot x_i \cdot \text{r1\_per\_unit}_i \leq 27380.54
\]
\[
\sum_{i \in I} 10 \cdot x_i \cdot \text{r2\_per\_unit}_i \leq 22245.11
\]
\[
\sum_{i \in I} 10 \cdot x_i \cdot \text{r3\_per\_unit}_i \leq 15147.73
\]

2. **Demand upper bound for each product**:

\[
10 \cdot x_i \leq \text{upper\_demand\_units}_i, \quad \forall i \in I
\]

3. **Batch integrality**:

\[
x_i \in \mathbb{Z}_+, \quad \forall i \in I
\]

---

##### Retrieved Parameters

- $I = \{P1, P2, \ldots, P100\}$
- For each $i \in I$:

| product | profit_per_unit | r1_per_unit | r2_per_unit | r3_per_unit | upper_demand_units | batch_size_units |
|---------|----------------|-------------|-------------|-------------|--------------------|------------------|
| P1      | 6.7            | 2.37        | 0.61        | 2.03        | 317                | 10               |
| P2      | 10.96          | 4.79        | 2.73        | 0.53        | 106                | 10               |
| P3      | 9.4            | 3.87        | 1.6         | 0.74        | 386                | 10               |
| P4      | 11.13          | 3.31        | 2.28        | 2.73        | 441                | 10               |
| P5      | 10.29          | 1.46        | 3.68        | 1.94        | 63                 | 10               |
| ...     | ...            | ...         | ...         | ...         | ...                | ...              |
| P100    | 8.84           | 1.25        | 3.23        | 0.53        | 576                | 10               |

- Resource capacities:
    - $C_1 = 27380.54$ (R1)
    - $C_2 = 22245.11$ (R2)
    - $C_3 = 15147.73$ (R3)

---

##### Complete Mathematical Model

\[
\begin{align*}
\max \quad & \sum_{i \in I} 10 \cdot x_i \cdot \text{profit\_per\_unit}_i \\
\text{s.t.} \quad
& \sum_{i \in I} 10 \cdot x_i \cdot \text{r1\_per\_unit}_i \leq 27380.54 \\
& \sum_{i \in I} 10 \cdot x_i \cdot \text{r2\_per\_unit}_i \leq 22245.11 \\
& \sum_{i \in I} 10 \cdot x_i \cdot \text{r3\_per\_unit}_i \leq 15147.73 \\
& 10 \cdot x_i \leq \text{upper\_demand\_units}_i, \quad \forall i \in I \\
& x_i \in \mathbb{Z}_+, \quad \forall i \in I
\end{align*}
\]

where all coefficients and bounds are as listed in the retrieved tables above.