##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods supplied from supplier $i \in I$ to supermarket $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is operational (open), 0 otherwise (binary).

##### Parameters

- $I = \{$S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12$\}$ (Suppliers)
- $J = \{$C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12$\}$ (Supermarkets)
- $d_j$: Demand of supermarket $j$ (from demand.csv)
- $f_i$: Fixed cost of opening supplier $i$ (from fixed_cost.csv)
- $c_{ij}$: Transportation cost per unit from supplier $i$ to supermarket $j$ (from transportation_costs.csv)
- $M = \sum_{j \in J} d_j = 2287$ (sufficiently large upper bound for each supplier's total shipment)

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

#### Data Mapping

- **Suppliers ($I$):** S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12
- **Supermarkets ($J$):** C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12

- **Demand $d_j$** (from demand.csv, column: "demand"):
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

- **Fixed cost $f_i$** (from fixed_cost.csv, column: "fixed_costs"):
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

- **Transportation cost $c_{ij}$** (from transportation_costs.csv, columns: "C1"–"C12", rows: "S1"–"S12"):
  - $c_{ij}$ is the entry in row $i$ (supplier) and column $j$ (supermarket).

- **Big-M constant:** $M = 2287$ (sum of all demands)

---

#### Source-Column Data Mapping

- demand.csv: customer $\to$ $j$, demand $\to$ $d_j$
- fixed_cost.csv: Unnamed: 0 $\to$ $i$, fixed_costs $\to$ $f_i$
- transportation_costs.csv: Unnamed: 0 $\to$ $i$, C1–C12 $\to$ $c_{ij}$

---

**Model summary:**  
Minimize total fixed and transportation costs by choosing which suppliers to open and how much each supermarket sources from each supplier, subject to demand fulfillment and supplier activation logic. All data and indices are mapped directly from the provided CSV columns.