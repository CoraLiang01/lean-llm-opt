##### Decision Variables

$x_{ij} \geq 0$: quantity of goods supplied from supplier $i \in I$ to supermarket $j \in J$ (continuous).
$y_i \in \{0,1\}$: whether supplier $i$ is operational (open).

##### Parameters

- $I = \{$S1, S2, ..., S24$\}$: set of suppliers (facilities).
- $J = \{$C1, C2, ..., C25$\}$: set of supermarkets (customers).
- $d_j$: demand of supermarket $j \in J$ (from demand.csv).
- $f_i$: fixed cost of opening supplier $i \in I$ (from fixed_cost.csv).
- $c_{ij}$: per-unit transportation cost from supplier $i$ to supermarket $j$ (from transportation_costs.csv).
- $M = \sum_{j \in J} d_j = 3258$: a valid upper bound for total supply from any supplier (since there are no explicit supplier capacity limits).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each supermarket $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]
2. **Supplier activation:**  
   For each supplier $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

---

#### Retrieved Parameters

- **Supermarkets (J) and Demands ($d_j$):**

| Customer | Demand |
|----------|--------|
| C1       | 1097   |
| C2       | 61     |
| C3       | 11     |
| C4       | 7      |
| C5       | 82     |
| C6       | 37     |
| C7       | 483    |
| C8       | 582    |
| C9       | 223    |
| C10      | 89     |
| C11      | 60     |
| C12      | 55     |
| C13      | 122    |
| C14      | 66     |
| C15      | 12     |
| C16      | 21     |
| C17      | 53     |
| C18      | 105    |
| C19      | 1      |
| C20      | 253    |
| C21      | 10     |
| C22      | 53     |
| C23      | 24     |
| C24      | 122    |
| C25      | 42     |

- **Suppliers (I) and Fixed Costs ($f_i$):**

| Supplier | Fixed Cost |
|----------|------------|
| S1       | 98.88      |
| S2       | 99.73      |
| S3       | 94.01      |
| S4       | 93.77      |
| S5       | 107.59     |
| S6       | 112.65     |
| S7       | 97.05      |
| S8       | 103        |
| S9       | 90.45      |
| S10      | 96.73      |
| S11      | 96.43      |
| S12      | 112.19     |
| S13      | 102.58     |
| S14      | 88.85      |
| S15      | 82.57      |
| S16      | 91.65      |
| S17      | 101.38     |
| S18      | 102.59     |
| S19      | 105.97     |
| S20      | 85.31      |
| S21      | 104.52     |
| S22      | 100.2      |
| S23      | 103.79     |
| S24      | 105.27     |

- **Transportation Costs ($c_{ij}$):**  
  (Matrix, rows: S1–S24, columns: C1–C25, all values as in transportation_costs.csv. For brevity, see above for full matrix.)

- **$M = 3258$** (sum of all demands).

---

##### Complete Mathematical Model

\[
\begin{align*}
\min\ & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.}\quad
& \sum_{i \in I} x_{ij} = d_j \qquad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq 3258\, y_i \qquad \forall i \in I \\
& x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\} \qquad \forall i \in I
\end{align*}
\]

Where all parameters ($d_j$, $f_i$, $c_{ij}$) are as retrieved above, with all identifiers and values preserved.