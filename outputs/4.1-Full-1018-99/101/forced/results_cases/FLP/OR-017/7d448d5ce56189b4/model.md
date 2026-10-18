##### Sets and Indices

Let $I$ be the set of all products in category 'ZZ', indexed by $i$.

##### Parameters

For each product $i \in I$:

- $r_i$: revenue per unit of product $i$ (from 'Revenue' column)
- $d_i$: demand for product $i$ (from 'Demand' column)
- $s_i$: initial inventory of product $i$ (from 'Initial Inventory' column)

The data for all $i \in I$ is as follows (partial list for illustration; full data as retrieved):

| SKU     | Revenue | Demand | Initial Inventory |
|---------|---------|--------|------------------|
| 00GVC   | 17.68   | 4      | 20.0             |
| 00OK1   | 1.27    | 33     | 180.0            |
| 0121I   | 2.03    | 60     | 310.0            |
| 01IEO   | 4.96    | 81     | 430.0            |
| 01IQT   | 1.5     | 14     | 70.0             |
| ...     | ...     | ...    | ...              |
| ZZX6K   | 111.81  | 2      | 10.0             |

(Full list as retrieved above.)

##### Decision Variables

$x_i \geq 0$: number of units of product $i$ to fulfill, for all $i \in I$ (continuous or integer, as appropriate).

##### Objective Function

\[
\max \sum_{i \in I} r_i x_i
\]

##### Constraints

1. Demand fulfillment cannot exceed demand:
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
2. Inventory cannot be exceeded:
   \[
   x_i \leq s_i, \quad \forall i \in I
   \]
3. Nonnegativity:
   \[
   x_i \geq 0, \quad \forall i \in I
   \]

##### Model Summary

\[
\begin{align*}
\max \quad & \sum_{i \in I} r_i x_i \\
\text{s.t.} \quad & x_i \leq d_i, \quad \forall i \in I \\
                  & x_i \leq s_i, \quad \forall i \in I \\
                  & x_i \geq 0, \quad \forall i \in I
\end{align*}
\]

where

- $I$ = set of all 'ZZ' products (full list above)
- $r_i$ = revenue per unit for product $i$ (from 'Revenue')
- $d_i$ = demand for product $i$ (from 'Demand')
- $s_i$ = initial inventory for product $i$ (from 'Initial Inventory')

##### Retrieved Data

All products in category 'ZZ' with their SKU, Revenue, Demand, and Initial Inventory, as listed above.