Let $I = \{A1, A2, \ldots, A80\}$ index the 80 products.

Let $x_i$ = number of 100 kg units of product $i$ produced in the month (integer, $\geq 0$).

Let $y_i$ = 1 if product $i$'s production line is activated, 0 otherwise (binary).

Parameters (for each $i \in I$):

- $D_i$ = Maximum Demand (100 kg units)
- $P_i$ = Selling Price ($/100 kg$)
- $C_i$ = Production Cost ($/100 kg$)
- $Q_i$ = Production Quota (max per day, 100 kg units)
- $F_i$ = Activation Cost ($)
- $B_i$ = Minimum Batch Size (100 kg units)
- $T = 22$ (number of production days in the month)

Data (from CSVs, in source order):

From 36-1.csv:

| Product | Maximum Demand (100 kg units) | Selling Price ($/100 kg) | Production Cost ($/100 kg) | Production Quota (max per day) |
|---------|------------------------------|--------------------------|----------------------------|-------------------------------|
| A1      | 5732                         | 111                      | 71.38                      | 586                           |
| A2      | 5607                         | 81                       | 45.02                      | 329                           |
| ...     | ...                          | ...                      | ...                        | ...                           |
| A80     | 4466                         | 106                      | 67.19                      | 556                           |

From 36-2.csv:

| Product | Activation Cost ($) |
|---------|---------------------|
| A1      | 178539              |
| A2      | 157708              |
| ...     | ...                 |
| A80     | 146842              |

From 36-3.csv:

| Product | Minimum Batch Size (100 kg units) |
|---------|-----------------------------------|
| A1      | 18                                |
| A2      | 25                                |
| ...     | ...                               |
| A80     | 18                                |

Model:

**Objective:**
\[
\max \sum_{i \in I} \left[ (P_i - C_i) x_i - F_i y_i \right]
\]

**Subject to:**

1. **Demand constraints:**
   \[
   x_i \leq D_i \qquad \forall i \in I
   \]

2. **Monthly production quota constraints:**
   \[
   x_i \leq Q_i \cdot T \qquad \forall i \in I
   \]

3. **Minimum batch size and activation logic:**
   \[
   x_i \geq B_i y_i \qquad \forall i \in I
   \]
   \[
   x_i \leq (Q_i \cdot T) y_i \qquad \forall i \in I
   \]
   (This ensures $x_i = 0$ if $y_i = 0$, and $x_i \geq B_i$ if $y_i = 1$.)

4. **Variable domains:**
   \[
   x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
   \]
   \[
   y_i \in \{0,1\} \qquad \forall i \in I
   \]

**Parameters (full data, source order):**

- $D_i$ (Maximum Demand), $P_i$ (Selling Price), $C_i$ (Production Cost), $Q_i$ (Production Quota), $F_i$ (Activation Cost), $B_i$ (Minimum Batch Size) as given in the CSVs above for each $i = $ A1, ..., A80.
- $T = 22$ (number of production days in the month).

**Decision variables:**

- $x_i$: integer, $\geq 0$ (number of 100 kg units of product $i$ produced in the month)
- $y_i$: binary (1 if product $i$'s production line is activated, 0 otherwise)

**All coefficients and identifiers are as retrieved and preserved in source order.**