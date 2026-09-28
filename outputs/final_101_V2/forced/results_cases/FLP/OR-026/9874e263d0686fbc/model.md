##### Decision Variables

Let $x_i$ be the number of units of Fashion product $i$ to fulfill, for each $i \in \mathcal{F}$ (the set of all Fashion products listed below).

##### Parameters

For each Fashion product $i$:
- $r_i$: Revenue per unit (from 'Revenue' column)
- $s_i$: Initial Inventory (from 'Initial Inventory' column)
- $d_i$: Demand (from 'Demand' column)

The set of Fashion products $\mathcal{F}$ and their parameters are:

| Product Name                   | $r_i$ | $s_i$ | $d_i$ |
|------------------------------- |-------|-------|-------|
| Fashion accessories_10.18      | 10.18 | 80    | 12    |
| Fashion accessories_12.09      | 12.09 | 10    | 2     |
| Fashion accessories_12.19      | 12.19 | 80    | 10    |
| Fashion accessories_12.54      | 12.54 | 10    | 2     |
| Fashion accessories_12.78      | 12.78 | 10    | 2     |
| Fashion accessories_14.48      | 14.48 | 40    | 5     |
| Fashion accessories_15.43      | 15.43 | 10    | 2     |
| Fashion accessories_15.5       | 15.5  | 10    | 2     |
| Fashion accessories_15.62      | 15.62 | 80    | 11    |
| Fashion accessories_16.28      | 16.28 | 10    | 2     |
| Fashion accessories_16.45      | 16.45 | 40    | 6     |
| Fashion accessories_17.48      | 17.48 | 60    | 9     |
| Fashion accessories_17.49      | 17.49 | 100   | 14    |
| Fashion accessories_17.87      | 17.87 | 40    | 6     |
| Fashion accessories_17.94      | 17.94 | 50    | 8     |
| Fashion accessories_18.08      | 18.08 | 40    | 6     |
| Fashion accessories_19.66      | 19.66 | 100   | 14    |
| Fashion accessories_19.7       | 19.7  | 10    | 2     |
| Fashion accessories_19.77      | 19.77 | 100   | 14    |
| Fashion accessories_20.01      | 20.01 | 90    | 11    |
| Fashion accessories_21.32      | 21.32 | 10    | 2     |
| Fashion accessories_21.48      | 21.48 | 20    | 3     |
| Fashion accessories_21.94      | 21.94 | 50    | 8     |
| Fashion accessories_22.32      | 22.32 | 80    | 10    |
| Fashion accessories_22.51      | 22.51 | 70    | 10    |
| Fashion accessories_23.82      | 23.82 | 50    | 7     |
| Fashion accessories_25.42      | 25.42 | 80    | 11    |
| Fashion accessories_25.56      | 25.56 | 70    | 11    |
| Fashion accessories_27.02      | 27.02 | 30    | 5     |
| Fashion accessories_27.18      | 27.18 | 20    | 3     |
| Fashion accessories_27.38      | 27.38 | 60    | 8     |
| Fashion accessories_29.42      | 29.42 | 100   | 13    |
| Fashion accessories_29.56      | 29.56 | 50    | 7     |
| Fashion accessories_30.14      | 30.14 | 100   | 14    |
| Fashion accessories_30.37      | 30.37 | 30    | 4     |
| Fashion accessories_30.61      | 30.61 | 10    | 2     |
| Fashion accessories_30.62      | 30.62 | 10    | 2     |
| Fashion accessories_31.73      | 31.73 | 90    | 14    |
| Fashion accessories_31.9       | 31.9  | 10    | 2     |
| Fashion accessories_32.62      | 32.62 | 40    | 6     |
| Fashion accessories_33.52      | 33.52 | 10    | 2     |
| Fashion accessories_33.63      | 33.63 | 10    | 2     |
| Fashion accessories_34.7       | 34.7  | 20    | 3     |
| Fashion accessories_35.19      | 35.19 | 100   | 14    |
| Fashion accessories_36.51      | 36.51 | 90    | 12    |
| Fashion accessories_36.85      | 36.85 | 50    | 7     |
| Fashion accessories_37.15      | 37.15 | 40    | 6     |
| Fashion accessories_37.55      | 37.55 | 100   | 13    |
| Fashion accessories_37.95      | 37.95 | 100   | 14    |

##### Objective Function

\[
\max \sum_{i \in \mathcal{F}} r_i x_i
\]

##### Constraints

1. Inventory and demand limits for each Fashion product $i$:
   \[
   0 \leq x_i \leq \min\{s_i, d_i\}, \quad \forall i \in \mathcal{F}
   \]

2. $x_i$ are integer variables (if only whole units can be fulfilled), or continuous if partial fulfillment is allowed.

##### Summary of Sets and Parameters

- $\mathcal{F}$: Set of all Fashion products listed above.
- $r_i$: Revenue per unit for product $i$ (see table).
- $s_i$: Initial Inventory for product $i$ (see table).
- $d_i$: Demand for product $i$ (see table).

##### Complete Model

\[
\begin{align*}
\max \quad & \sum_{i \in \mathcal{F}} r_i x_i \\
\text{s.t.} \quad & 0 \leq x_i \leq \min\{s_i, d_i\}, \quad \forall i \in \mathcal{F} \\
& x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{F} \quad \text{(if integer units required)}
\end{align*}
\]

All parameter values are as listed above.