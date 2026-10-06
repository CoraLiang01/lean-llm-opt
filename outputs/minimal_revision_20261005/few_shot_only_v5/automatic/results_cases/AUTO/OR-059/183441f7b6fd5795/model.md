##### Decision Variables

- $x_{ij} \geq 0$: Number of vehicles shipped from supplier $i \in I$ to dealership $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is operational (open), 0 otherwise (binary).

##### Parameters

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}\}$ (Suppliers)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}\}$ (Dealerships)
- $d_j$: Demand of dealership $j \in J$
- $f_i$: Fixed cost to open supplier $i \in I$
- $c_{ij}$: Transportation cost per vehicle from supplier $i$ to dealership $j$
- $M = \sum_{j \in J} d_j$ (sufficiently large upper bound for each supplier's total shipment)

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each dealership $j \in J$,
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

- **Suppliers ($I$):** S1, S2, S3, S4, S5, S6, S7, S8  
  (from fixed_cost.csv, Unnamed: 0 column)
- **Dealerships ($J$):** C1, C2, C3, C4, C5, C6, C7, C8, C9  
  (from demand.csv, customer column)

- **Demand ($d_j$):**  
  (from demand.csv, demand column)
  - C1: 4,742,532,000
  - C2: 1,600,594,000
  - C3: 5,086,889,000
  - C4: 1,027,326,000
  - C5: 11,926,044,000
  - C6: 9,058,407,000
  - C7: 5,344,367,000
  - C8: 677,201,000
  - C9: 3,236,493,000

- **Fixed costs ($f_i$):**  
  (from fixed_cost.csv, fixed_costs column)
  - S1: 100.64
  - S2: 98.72
  - S3: 100.18
  - S4: 96.58
  - S5: 95.75
  - S6: 99.06
  - S7: 101.78
  - S8: 93.86

- **Transportation costs ($c_{ij}$):**  
  (from transportation_costs.csv, Unnamed: 0 as supplier, columns C1–C9 as dealerships)

  |        |  C1    |   C2    |   C3    |   C4    |   C5    |   C6    |   C7    |   C8    |   C9    |
  |--------|--------|---------|---------|---------|---------|---------|---------|---------|---------|
  | **S1** | 1091.04| 85.72   | 99.08   | 747.35  | 893.86  | 23.65   | 15.11   | 15.03   | 497.88  |
  | **S2** | 58.88  | 1617.16 | 1786.44 | 951.81  | 56.45   | 642.77  | 16.69   | 0.63    | 11.2    |
  | **S3** | 110.47 | 0.04    | 38.89   | 1397.95 | 2361.45 | 107.62  | 1598.5  | 76.41   | 1382.84 |
  | **S4** | 1458.85| 1049.27 | 597.32  | 1731.9  | 69.09   | 1227.17 | 1187.55 | 1017.16 | 52.15   |
  | **S5** | 0.38   | 2315.52 | 1313.06 | 1253.71 | 50.24   | 29.19   | 60.17   | 1077.35 | 70.11   |
  | **S6** | 58.2   | 1395.81 | 84.6    | 830.64  | 1003.86 | 631.17  | 31.13   | 1.4     | 246.24  |
  | **S7** | 1255.23| 1382.31 | 78.79   | 829.02  | 67.31   | 877.35  | 185.28  | 221.98  | 0.05    |
  | **S8** | 1990.09| 1.23    | 38.97   | 1396.35 | 112.54  | 107.54  | 1596.74 | 76.32   | 1183.79 |

- **Big-M ($M$):**  
  $M = 4,742,532,000 + 1,600,594,000 + 5,086,889,000 + 1,027,326,000 + 11,926,044,000 + 9,058,407,000 + 5,344,367,000 + 677,201,000 + 3,236,493,000 = 42,700,853,000$

---

#### Source-Column Data Mapping

- demand.csv: customer → $j$, demand → $d_j$
- fixed_cost.csv: Unnamed: 0 → $i$, fixed_costs → $f_i$
- transportation_costs.csv: Unnamed: 0 → $i$, C1–C9 → $c_{ij}$

---

**Model summary:**  
Minimize total fixed and transportation costs by selecting which suppliers to open and how much each dealership sources from each supplier, subject to demand fulfillment and supplier activation logic. All parameters and sets are mapped directly from the provided CSV columns.