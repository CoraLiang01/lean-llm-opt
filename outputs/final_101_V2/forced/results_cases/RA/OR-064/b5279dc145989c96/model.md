Let $F$ be the set of suppliers (indexed by $i$), and $C$ the set of supermarkets/customers (indexed by $j$).

**Parameters:**

- $f_i$: fixed cost of opening supplier $i$ (from fixed_cost.csv)
- $t_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from transportation_costs.csv)
- $d_j$: demand of customer $j$ (from demand.csv)

**Decision Variables:**

- $y_i \in \{0,1\}$: 1 if supplier $i$ is open, 0 otherwise
- $x_{ij} \geq 0$: quantity supplied from supplier $i$ to customer $j$

---

**Objective:**

$$
\min \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in C} t_{ij} x_{ij}
$$

---

**Constraints:**

1. **Demand Satisfaction:**

   For each customer $j \in C$:
   $$
   \sum_{i \in F} x_{ij} = d_j
   $$

2. **Supply Only from Open Suppliers:**

   For all $i \in F$, $j \in C$:
   $$
   x_{ij} \leq d_j y_i
   $$

3. **Variable Domains:**

   $$
   y_i \in \{0,1\} \quad \forall i \in F
   $$
   $$
   x_{ij} \geq 0 \quad \forall i \in F,\, j \in C
   $$

---

**Data (as retrieved):**

- **Suppliers $F$ (from fixed_cost.csv, Unnamed: 0):**
  S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17, S18, S19, S20, S21, S22, S23, S24

- **Fixed costs $f_i$ (from fixed_cost.csv):**
  - S1: 98.88
  - S2: 99.73
  - S3: 94.01
  - S4: 93.77
  - S5: 107.59
  - S6: 112.65
  - S7: 97.05
  - S8: 103
  - S9: 90.45
  - S10: 96.73
  - S11: 96.43
  - S12: 112.19
  - S13: 102.58
  - S14: 88.85
  - S15: 82.57
  - S16: 91.65
  - S17: 101.38
  - S18: 102.59
  - S19: 105.97
  - S20: 85.31
  - S21: 104.52
  - S22: 100.2
  - S23: 103.79
  - S24: 105.27

- **Supermarkets/Customers $C$ (from transportation_costs.csv columns, and demand.csv):**
  C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15, C16, C17, C18, C19, C20, C21, C22, C23, C24, C25

- **Transportation costs $t_{ij}$ (from transportation_costs.csv):**
  For each supplier $i$ (row Unnamed: 0), and each customer $j$ (columns C1–C25), use the exact values as given in the file.

- **Demands $d_j$ (from demand.csv):**
  - C1: 1097
  - C2: 61
  - C3: 11
  - C4: 7
  - C5: 82
  - C6: 37
  - C7: 483
  - C8: 582
  - C9: 223
  - C10: 89
  - C11: 60
  - C12: 55
  - C13: 122
  - C14: 66
  - C15: 12
  - C16: 21
  - C17: 53
  - C18: 105
  - C19: 1
  - C20: 253
  - C21: 10
  - C22: 53
  - C23: 24
  - C24: 122
  - C25: 42

---

**Complete Model:**

$$
\begin{align*}
\min \quad & \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in C} t_{ij} x_{ij} \\
\text{s.t.} \quad & \sum_{i \in F} x_{ij} = d_j \quad \forall j \in C \\
& x_{ij} \leq d_j y_i \quad \forall i \in F,\, j \in C \\
& y_i \in \{0,1\} \quad \forall i \in F \\
& x_{ij} \geq 0 \quad \forall i \in F,\, j \in C \\
\end{align*}
$$

where all indices, costs, and demands are as listed above, and all data is used exactly as retrieved.