##### Decision Variables

$x_i \in \mathbb{Z}_{\geq 0}$: Number of units of product $i$ to fulfill, for each $i \in I$ (where $I$ is the set of all products with IDs starting with 'S700_').

##### Parameters

Let $I = \{$S700_1138, S700_1691, S700_1938, S700_2047, S700_2466, S700_2610, S700_2824, S700_2834, S700_3167, S700_3505, S700_3962, S700_4002$\}$.

For each $i \in I$:

- $\text{Revenue}_i$:
  - S700_1138: 70.67
  - S700_1691: 100.0
  - S700_1938: 70.15
  - S700_2047: 100.0
  - S700_2466: 100.0
  - S700_2610: 65.77
  - S700_2824: 100.0
  - S700_2834: 100.0
  - S700_3167: 74.4
  - S700_3505: 81.14
  - S700_3962: 100.0
  - S700_4002: 61.44

- $\text{InitialInventory}_i$:
  - S700_1138: 9020
  - S700_1691: 8370
  - S700_1938: 8390
  - S700_2047: 8680
  - S700_2466: 9400
  - S700_2610: 9900
  - S700_2824: 9760
  - S700_2834: 8610
  - S700_3167: 9380
  - S700_3505: 9170
  - S700_3962: 8520
  - S700_4002: 10290

- $\text{Demand}_i$:
  - S700_1138: 1219
  - S700_1691: 1127
  - S700_1938: 1129
  - S700_2047: 1176
  - S700_2466: 1301
  - S700_2610: 1340
  - S700_2824: 1357
  - S700_2834: 1158
  - S700_3167: 1287
  - S700_3505: 1281
  - S700_3962: 1135
  - S700_4002: 1392

##### Objective Function

\[
\max \sum_{i \in I} \text{Revenue}_i \cdot x_i
\]

##### Constraints

For each $i \in I$:

1. Inventory and demand limits:
   \[
   0 \leq x_i \leq \min\{\text{InitialInventory}_i, \text{Demand}_i\}
   \]
   (with $x_i$ integer)

##### Explicitly, for each product:

\[
\begin{align*}
0 \leq x_{\text{S700\_1138}} &\leq 1219 \\
0 \leq x_{\text{S700\_1691}} &\leq 1127 \\
0 \leq x_{\text{S700\_1938}} &\leq 1129 \\
0 \leq x_{\text{S700\_2047}} &\leq 1176 \\
0 \leq x_{\text{S700\_2466}} &\leq 1301 \\
0 \leq x_{\text{S700\_2610}} &\leq 1340 \\
0 \leq x_{\text{S700\_2824}} &\leq 1357 \\
0 \leq x_{\text{S700\_2834}} &\leq 1158 \\
0 \leq x_{\text{S700\_3167}} &\leq 1287 \\
0 \leq x_{\text{S700\_3505}} &\leq 1281 \\
0 \leq x_{\text{S700\_3962}} &\leq 1135 \\
0 \leq x_{\text{S700\_4002}} &\leq 1392 \\
\end{align*}
\]

##### Variable Domains

\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\]

##### Summary Table

| Product      | Revenue | Initial Inventory | Demand | $x_i$ bounds         |
|--------------|---------|------------------|--------|----------------------|
| S700_1138    | 70.67   | 9020             | 1219   | $0 \leq x \leq 1219$ |
| S700_1691    | 100.0   | 8370             | 1127   | $0 \leq x \leq 1127$ |
| S700_1938    | 70.15   | 8390             | 1129   | $0 \leq x \leq 1129$ |
| S700_2047    | 100.0   | 8680             | 1176   | $0 \leq x \leq 1176$ |
| S700_2466    | 100.0   | 9400             | 1301   | $0 \leq x \leq 1301$ |
| S700_2610    | 65.77   | 9900             | 1340   | $0 \leq x \leq 1340$ |
| S700_2824    | 100.0   | 9760             | 1357   | $0 \leq x \leq 1357$ |
| S700_2834    | 100.0   | 8610             | 1158   | $0 \leq x \leq 1158$ |
| S700_3167    | 74.4    | 9380             | 1287   | $0 \leq x \leq 1287$ |
| S700_3505    | 81.14   | 9170             | 1281   | $0 \leq x \leq 1281$ |
| S700_3962    | 100.0   | 8520             | 1135   | $0 \leq x \leq 1135$ |
| S700_4002    | 61.44   | 10290            | 1392   | $0 \leq x \leq 1392$ |