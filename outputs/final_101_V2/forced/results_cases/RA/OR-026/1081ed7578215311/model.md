Let $I$ be the set of all products listed below, each with its associated data. For each product $i \in I$, let:

- $x_i$ = number of units of product $i$ to fulfill (decision variable, nonnegative integer)
- $r_i$ = revenue per unit of product $i$ (from ‘Revenue’ column)
- $d_i$ = demand for product $i$ (from ‘Demand’ column)
- $s_i$ = initial inventory for product $i$ (from ‘Initial Inventory’ column)

#### Sets and Parameters

| Product Name                       | $r_i$ | $d_i$ | $s_i$ |
|-------------------------------------|-------|-------|-------|
| Fashion accessories_10.18           | 10.18 | 12    | 80    |
| Fashion accessories_12.09           | 12.09 | 2     | 10    |
| Fashion accessories_12.19           | 12.19 | 10    | 80    |
| Fashion accessories_12.54           | 12.54 | 2     | 10    |
| Fashion accessories_12.78           | 12.78 | 2     | 10    |
| Fashion accessories_14.48           | 14.48 | 5     | 40    |
| Fashion accessories_15.43           | 15.43 | 2     | 10    |
| Fashion accessories_15.5            | 15.5  | 2     | 10    |
| Fashion accessories_15.62           | 15.62 | 11    | 80    |
| Fashion accessories_16.28           | 16.28 | 2     | 10    |
| Fashion accessories_16.45           | 16.45 | 6     | 40    |
| Fashion accessories_17.48           | 17.48 | 9     | 60    |
| Fashion accessories_17.49           | 17.49 | 14    | 100   |
| Fashion accessories_17.87           | 17.87 | 6     | 40    |
| Fashion accessories_17.94           | 17.94 | 8     | 50    |
| Fashion accessories_18.08           | 18.08 | 6     | 40    |
| Fashion accessories_19.66           | 19.66 | 14    | 100   |
| Fashion accessories_19.7            | 19.7  | 2     | 10    |
| Fashion accessories_19.77           | 19.77 | 14    | 100   |
| Fashion accessories_20.01           | 20.01 | 11    | 90    |
| Fashion accessories_21.32           | 21.32 | 2     | 10    |
| Fashion accessories_21.48           | 21.48 | 3     | 20    |
| Fashion accessories_21.94           | 21.94 | 8     | 50    |
| Fashion accessories_22.32           | 22.32 | 10    | 80    |
| Fashion accessories_22.51           | 22.51 | 10    | 70    |
| Fashion accessories_23.82           | 23.82 | 7     | 50    |
| Fashion accessories_25.42           | 25.42 | 11    | 80    |
| Fashion accessories_25.56           | 25.56 | 11    | 70    |
| Fashion accessories_27.02           | 27.02 | 5     | 30    |
| Fashion accessories_27.18           | 27.18 | 3     | 20    |
| Fashion accessories_27.38           | 27.38 | 8     | 60    |
| Fashion accessories_29.42           | 29.42 | 13    | 100   |
| Fashion accessories_29.56           | 29.56 | 7     | 50    |
| Fashion accessories_30.14           | 30.14 | 14    | 100   |
| Fashion accessories_30.37           | 30.37 | 4     | 30    |
| Fashion accessories_30.61           | 30.61 | 2     | 10    |
| Fashion accessories_30.62           | 30.62 | 2     | 10    |
| Fashion accessories_31.73           | 31.73 | 14    | 90    |
| Fashion accessories_31.9            | 31.9  | 2     | 10    |
| Fashion accessories_32.62           | 32.62 | 6     | 40    |
| Fashion accessories_33.52           | 33.52 | 2     | 10    |
| Fashion accessories_33.63           | 33.63 | 2     | 10    |
| Fashion accessories_34.7            | 34.7  | 3     | 20    |
| Fashion accessories_35.19           | 35.19 | 14    | 100   |
| Fashion accessories_36.51           | 36.51 | 12    | 90    |
| Fashion accessories_36.85           | 36.85 | 7     | 50    |
| Fashion accessories_37.15           | 37.15 | 6     | 40    |
| Fashion accessories_37.55           | 37.55 | 13    | 100   |
| Fashion accessories_37.95           | 37.95 | 14    | 100   |

#### Decision Variables

For each product $i$ in the table above:
$$
x_i \in \mathbb{Z}_{\geq 0}
$$

#### Objective

Maximize total revenue from fulfilled Fashion product demand:
$$
\max \sum_{i \in I} r_i x_i
$$

#### Constraints

For each product $i \in I$:

1. Cannot fulfill more than demand:
   $$
   x_i \leq d_i
   $$
2. Cannot fulfill more than available inventory:
   $$
   x_i \leq s_i
   $$
3. Nonnegativity and integrality:
   $$
   x_i \in \mathbb{Z}_{\geq 0}
   $$

#### Complete Model

$$
\begin{align*}
\max \quad & \sum_{i \in I} r_i x_i \\
\text{s.t.} \quad & x_i \leq d_i, \quad \forall i \in I \\
                  & x_i \leq s_i, \quad \forall i \in I \\
                  & x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\end{align*}
$$

Where $I$ and all parameters are as listed in the table above.