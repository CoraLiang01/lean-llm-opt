##### Sets and Indices

Let $I$ be the set of products with identifiers starting with ‘S700_’:
$$
I = \{\text{S700\_1138},\ \text{S700\_1691},\ \text{S700\_1938},\ \text{S700\_2047},\ \text{S700\_2466},\ \text{S700\_2610},\ \text{S700\_2824},\ \text{S700\_2834},\ \text{S700\_3167},\ \text{S700\_3505},\ \text{S700\_3962},\ \text{S700\_4002}\}
$$

##### Parameters

For each $i \in I$:

- $r_i$: Revenue per unit of product $i$
- $d_i$: Demand for product $i$
- $s_i$: Initial inventory of product $i$

The parameter values are:

| Product         | $r_i$  | $d_i$ | $s_i$  |
|-----------------|--------|-------|--------|
| S700_1138       | 70.67  | 1219  | 9020   |
| S700_1691       | 100.0  | 1127  | 8370   |
| S700_1938       | 70.15  | 1129  | 8390   |
| S700_2047       | 100.0  | 1176  | 8680   |
| S700_2466       | 100.0  | 1301  | 9400   |
| S700_2610       | 65.77  | 1340  | 9900   |
| S700_2824       | 100.0  | 1357  | 9760   |
| S700_2834       | 100.0  | 1158  | 8610   |
| S700_3167       | 74.4   | 1287  | 9380   |
| S700_3505       | 81.14  | 1281  | 9170   |
| S700_3962       | 100.0  | 1135  | 8520   |
| S700_4002       | 61.44  | 1392  | 10290  |

##### Decision Variables

For each $i \in I$:

- $x_i \geq 0$: Number of units of product $i$ to fulfill (integer or continuous, as appropriate).

##### Objective Function

Maximize total revenue:
$$
\max \sum_{i \in I} r_i x_i
$$

##### Constraints

1. Demand and inventory limits:
   $$
   0 \leq x_i \leq \min\{d_i,\, s_i\}, \quad \forall i \in I
   $$

##### Complete Model

\[
\begin{align*}
\max\quad & \sum_{i \in I} r_i x_i \\
\text{s.t.}\quad & 0 \leq x_i \leq \min\{d_i,\, s_i\}, \quad \forall i \in I \\
\end{align*}
\]

Where the parameters are as listed above.