##### Decision Variables

- $x_{ij} \geq 0$: Quantity of Adidas products shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Store demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Supplier activation constraint:**  
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j = 1080$.
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}\}$ (Suppliers)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}\}$ (Stores)

##### Parameters

- **Store demands $d_j$:**

| Store | Demand |
|-------|--------|
| C1    | 216    |
| C2    | 216    |
| C3    | 216    |
| C4    | 144    |
| C5    | 144    |
| C6    | 144    |

- **Supplier fixed costs $f_i$:**

| Supplier | Fixed Cost |
|----------|------------|
| S1       | 98.88      |
| S2       | 99.73      |
| S3       | 94.01      |
| S4       | 93.77      |
| S5       | 107.59     |
| S6       | 112.65     |

- **Transportation costs $c_{ij}$ (per unit from supplier $i$ to store $j$):**

|        |   C1   |   C2   |   C3   |   C4   |   C5   |   C6   |
|--------|--------|--------|--------|--------|--------|--------|
| **S1** |  0.08  | 52.33  | 73.57  |1237.33 | 0.07   |112.16  |
| **S2** | 46.02  |175.23  |2026.83 |299.89  |966.53  |1590.42 |
| **S3** |1031.74 | 78.13  | 99.02  |277.07  |884.45  |1800.86 |
| **S4** | 868.75 | 94.20  |1776.34 |285.48  |868.85  | 86.55  |
| **S5** |1577.00 |760.15  |2090.19 | 43.20  |1577.12 |1095.17 |
| **S6** | 49.14  |  4.33  |2079.57 |277.04  |1032.01 |1543.49 |

- **Big-M parameter:** $M = 216 + 216 + 216 + 144 + 144 + 144 = 1080$

##### Complete Model

\[
\begin{align*}
\min \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.} \quad & \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I \\
& x_{ij} \geq 0, \quad \forall i \in I, j \in J \\
& y_i \in \{0,1\}, \quad \forall i \in I
\end{align*}
\]

where all parameters are as listed above.