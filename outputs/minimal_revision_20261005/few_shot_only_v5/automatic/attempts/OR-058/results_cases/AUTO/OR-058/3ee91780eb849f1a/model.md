##### Decision Variables

- $x_{ij} \geq 0$: Quantity of Adidas products shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is operational (open), 0 otherwise (binary).

##### Parameters

- $I = \{S1, S2, S3, S4, S5, S6\}$: Set of suppliers.
- $J = \{C1, C2, C3, C4, C5, C6\}$: Set of stores.
- $d_j$: Demand at store $j$.
- $f_i$: Fixed cost to open supplier $i$.
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$.
- $M = \sum_{j \in J} d_j = 216 + 216 + 216 + 144 + 144 + 144 = 1080$: A valid upper bound for total shipments from any supplier.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction at each store:**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation:**
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0, \quad y_i \in \{0,1\}
   \]

---

#### Data Mapping

- **Suppliers ($I$):** S1, S2, S3, S4, S5, S6
- **Stores ($J$):** C1, C2, C3, C4, C5, C6

- **Demand ($d_j$):**  
  - C1: 216  
  - C2: 216  
  - C3: 216  
  - C4: 144  
  - C5: 144  
  - C6: 144  

- **Fixed Costs ($f_i$):**  
  - S1: 98.88  
  - S2: 99.73  
  - S3: 94.01  
  - S4: 93.77  
  - S5: 107.59  
  - S6: 112.65  

- **Transportation Costs ($c_{ij}$):**  
  (Rows: suppliers S1–S6; Columns: stores C1–C6)

  |        | C1     | C2     | C3      | C4      | C5      | C6      |
  |--------|--------|--------|---------|---------|---------|---------|
  | **S1** | 0.08   | 52.33  | 73.57   | 1237.33 | 0.07    | 112.16  |
  | **S2** | 46.02  | 175.23 | 2026.83 | 299.89  | 966.53  | 1590.42 |
  | **S3** | 1031.74| 78.13  | 99.02   | 277.07  | 884.45  | 1800.86 |
  | **S4** | 868.75 | 94.2   | 1776.34 | 285.48  | 868.85  | 86.55   |
  | **S5** | 1577   | 760.15 | 2090.19 | 43.2    | 1577.12 | 1095.17 |
  | **S6** | 49.14  | 4.33   | 2079.57 | 277.04  | 1032.01 | 1543.49 |

- **Big-M ($M$):** 1080

---

**Source-column Data Mapping:**

- demand.csv: customer → $j$, demand → $d_j$
- fixed_cost.csv: Unnamed: 0 → $i$, fixed_costs → $f_i$
- transportation_costs.csv: Unnamed: 0 → $i$, C1–C6 → $c_{ij}$

---

**Complete Model:**

\[
\begin{align*}
\min\ & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.}\quad
& \sum_{i \in I} x_{ij} = d_j, && \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq M y_i, && \forall i \in I \\
& x_{ij} \geq 0, \quad y_i \in \{0,1\}
\end{align*}
\]