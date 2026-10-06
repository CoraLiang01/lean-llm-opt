##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is activated, 0 otherwise (binary).

##### Parameters

- $I = \{S1, S2, S3, S4, S5, S6, S7\}$: Set of warehouses.
- $J = \{C1, C2, C3, C4, C5, C6, C7\}$: Set of musicians/bands.
- $d_j$: Demand of musician/band $j$.
- $f_i$: Fixed cost of activating warehouse $i$.
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to musician/band $j$.
- $M = \sum_{j \in J} d_j = 37,\!058$: A sufficiently large constant for linking constraints.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each musician/band $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]
2. **Warehouse activation:**  
   For each warehouse $i \in I$,
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

- **Warehouses ($I$):** S1, S2, S3, S4, S5, S6, S7  
  (from fixed_cost.csv and transportation_costs.csv, column "Unnamed: 0")
- **Musicians/Bands ($J$):** C1, C2, C3, C4, C5, C6, C7  
  (from demand.csv and transportation_costs.csv, columns "C1"..."C7")
- **Demand ($d_j$):**  
  - C1: 1083  
  - C2: 776  
  - C3: 16214  
  - C4: 553  
  - C5: 17106  
  - C6: 594  
  - C7: 732  
  (from demand.csv, columns "customer", "demand")
- **Fixed costs ($f_i$):**  
  - S1: 102.33  
  - S2: 94.92  
  - S3: 91.83  
  - S4: 98.71  
  - S5: 95.73  
  - S6: 99.96  
  - S7: 98.16  
  (from fixed_cost.csv, columns "Unnamed: 0", "fixed_costs")
- **Transportation costs ($c_{ij}$):**  
  (from transportation_costs.csv, rows "Unnamed: 0" = S1...S7, columns "C1"..."C7")

|        |  C1    |   C2   |   C3   |   C4   |   C5   |   C6   |   C7   |
|--------|--------|--------|--------|--------|--------|--------|--------|
| **S1** | 1506.22| 70.90  | 8.44   | 260.27 | 197.47 | 71.71  | 61.19  |
| **S2** | 1732.65| 1780.72| 567.44 | 448.68 | 29.00  | 1484.91| 963.92 |
| **S3** | 115.66 | 100.76 | 64.68  | 1324.53| 64.99  | 134.88 | 2102.83|
| **S4** | 1254.78| 1115.63| 52.31  | 1036.16| 892.63 | 1464.04| 1383.41|
| **S5** | 42.90  | 891.01 | 1013.94| 1128.72| 58.91  | 42.89  | 1570.31|
| **S6** | 0.70   | 139.46 | 70.03  | 79.15  | 1482.00| 0.91   | 110.46 |
| **S7** | 1732.30| 1780.44| 486.50 | 523.74 | 522.08 | 82.48  | 826.41 |

- **Big-M ($M$):** $M = 1083 + 776 + 16214 + 553 + 17106 + 594 + 732 = 37,\!058$

---

**Source-column Data Mapping:**  
- demand.csv: ("customer", "demand")  
- fixed_cost.csv: ("Unnamed: 0", "fixed_costs")  
- transportation_costs.csv: ("Unnamed: 0", "C1", ..., "C7")