##### Sets and Indices

Let $I$ be the set of all products classified as 'Fashion':

\[
I = \{
\text{Fashion accessories\_10.18},
\text{Fashion accessories\_12.09},
\text{Fashion accessories\_12.19},
\text{Fashion accessories\_12.54},
\text{Fashion accessories\_12.78},
\text{Fashion accessories\_14.48},
\text{Fashion accessories\_15.43},
\text{Fashion accessories\_15.5},
\text{Fashion accessories\_15.62},
\text{Fashion accessories\_16.28},
\text{Fashion accessories\_16.45},
\text{Fashion accessories\_17.48},
\text{Fashion accessories\_17.49},
\text{Fashion accessories\_17.87},
\text{Fashion accessories\_17.94},
\text{Fashion accessories\_18.08},
\text{Fashion accessories\_19.66},
\text{Fashion accessories\_19.7},
\text{Fashion accessories\_19.77},
\text{Fashion accessories\_20.01},
\text{Fashion accessories\_21.32},
\text{Fashion accessories\_21.48},
\text{Fashion accessories\_21.94},
\text{Fashion accessories\_22.32},
\text{Fashion accessories\_22.51},
\text{Fashion accessories\_23.82},
\text{Fashion accessories\_25.42},
\text{Fashion accessories\_25.56},
\text{Fashion accessories\_27.02},
\text{Fashion accessories\_27.18},
\text{Fashion accessories\_27.38},
\text{Fashion accessories\_29.42},
\text{Fashion accessories\_29.56},
\text{Fashion accessories\_30.14},
\text{Fashion accessories\_30.37},
\text{Fashion accessories\_30.61},
\text{Fashion accessories\_30.62},
\text{Fashion accessories\_31.73},
\text{Fashion accessories\_31.9},
\text{Fashion accessories\_32.62},
\text{Fashion accessories\_33.52},
\text{Fashion accessories\_33.63},
\text{Fashion accessories\_34.7},
\text{Fashion accessories\_35.19},
\text{Fashion accessories\_36.51},
\text{Fashion accessories\_36.85},
\text{Fashion accessories\_37.15},
\text{Fashion accessories\_37.55},
\text{Fashion accessories\_37.95}
\}
\]

##### Parameters

For each $i \in I$:

- $r_i$: Revenue per unit of product $i$
- $d_i$: Demand for product $i$
- $s_i$: Initial inventory of product $i$

The parameter values are:

| Product Name                   | $r_i$ | $d_i$ | $s_i$ |
|------------------------------- |-------|-------|-------|
| Fashion accessories_10.18      | 10.18 | 12    | 80    |
| Fashion accessories_12.09      | 12.09 | 2     | 10    |
| Fashion accessories_12.19      | 12.19 | 10    | 80    |
| Fashion accessories_12.54      | 12.54 | 2     | 10    |
| Fashion accessories_12.78      | 12.78 | 2     | 10    |
| Fashion accessories_14.48      | 14.48 | 5     | 40    |
| Fashion accessories_15.43      | 15.43 | 2     | 10    |
| Fashion accessories_15.5       | 15.5  | 2     | 10    |
| Fashion accessories_15.62      | 15.62 | 11    | 80    |
| Fashion accessories_16.28      | 16.28 | 2     | 10    |
| Fashion accessories_16.45      | 16.45 | 6     | 40    |
| Fashion accessories_17.48      | 17.48 | 9     | 60    |
| Fashion accessories_17.49      | 17.49 | 14    | 100   |
| Fashion accessories_17.87      | 17.87 | 6     | 40    |
| Fashion accessories_17.94      | 17.94 | 8     | 50    |
| Fashion accessories_18.08      | 18.08 | 6     | 40    |
| Fashion accessories_19.66      | 19.66 | 14    | 100   |
| Fashion accessories_19.7       | 19.7  | 2     | 10    |
| Fashion accessories_19.77      | 19.77 | 14    | 100   |
| Fashion accessories_20.01      | 20.01 | 11    | 90    |
| Fashion accessories_21.32      | 21.32 | 2     | 10    |
| Fashion accessories_21.48      | 21.48 | 3     | 20    |
| Fashion accessories_21.94      | 21.94 | 8     | 50    |
| Fashion accessories_22.32      | 22.32 | 10    | 80    |
| Fashion accessories_22.51      | 22.51 | 10    | 70    |
| Fashion accessories_23.82      | 23.82 | 7     | 50    |
| Fashion accessories_25.42      | 25.42 | 11    | 80    |
| Fashion accessories_25.56      | 25.56 | 11    | 70    |
| Fashion accessories_27.02      | 27.02 | 5     | 30    |
| Fashion accessories_27.18      | 27.18 | 3     | 20    |
| Fashion accessories_27.38      | 27.38 | 8     | 60    |
| Fashion accessories_29.42      | 29.42 | 13    | 100   |
| Fashion accessories_29.56      | 29.56 | 7     | 50    |
| Fashion accessories_30.14      | 30.14 | 14    | 100   |
| Fashion accessories_30.37      | 30.37 | 4     | 30    |
| Fashion accessories_30.61      | 30.61 | 2     | 10    |
| Fashion accessories_30.62      | 30.62 | 2     | 10    |
| Fashion accessories_31.73      | 31.73 | 14    | 90    |
| Fashion accessories_31.9       | 31.9  | 2     | 10    |
| Fashion accessories_32.62      | 32.62 | 6     | 40    |
| Fashion accessories_33.52      | 33.52 | 2     | 10    |
| Fashion accessories_33.63      | 33.63 | 2     | 10    |
| Fashion accessories_34.7       | 34.7  | 3     | 20    |
| Fashion accessories_35.19      | 35.19 | 14    | 100   |
| Fashion accessories_36.51      | 36.51 | 12    | 90    |
| Fashion accessories_36.85      | 36.85 | 7     | 50    |
| Fashion accessories_37.15      | 37.15 | 6     | 40    |
| Fashion accessories_37.55      | 37.55 | 13    | 100   |
| Fashion accessories_37.95      | 37.95 | 14    | 100   |

##### Decision Variables

For each $i \in I$:

- $x_i \geq 0$: Number of units of product $i$ to fulfill (continuous or integer, as appropriate for the context).

##### Objective Function

\[
\max \sum_{i \in I} r_i x_i
\]

##### Constraints

1. Inventory and demand bounds for each product:
   \[
   0 \leq x_i \leq \min\{d_i, s_i\}, \quad \forall i \in I
   \]

##### Complete Model

\[
\begin{align*}
\max \quad & \sum_{i \in I} r_i x_i \\
\text{s.t.} \quad & 0 \leq x_i \leq \min\{d_i, s_i\}, \quad \forall i \in I \\
& x_i \geq 0, \quad \forall i \in I
\end{align*}
\]

where all parameters and sets are as listed above.