#### Sets and Indices

- Let $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}\}$ be the set of suppliers, indexed by $i$.
- Let $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}\}$ be the set of dealerships (customers), indexed by $j$.

#### Parameters

- $f_i$: Fixed cost to open supplier $i$.

  - $f_{\text{S1}} = 100.64$
  - $f_{\text{S2}} = 98.72$
  - $f_{\text{S3}} = 100.18$
  - $f_{\text{S4}} = 96.58$
  - $f_{\text{S5}} = 95.75$
  - $f_{\text{S6}} = 99.06$
  - $f_{\text{S7}} = 101.78$
  - $f_{\text{S8}} = 93.86$

- $c_{ij}$: Transportation cost per vehicle from supplier $i$ to dealership $j$.

  |        |  C1     |   C2    |   C3    |   C4    |   C5    |   C6    |   C7    |   C8    |   C9    |
  |--------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
  | S1     | 1091.04 | 85.72   | 99.08   | 747.35  | 893.86  | 23.65   | 15.11   | 15.03   | 497.88  |
  | S2     | 58.88   | 1617.16 | 1786.44 | 951.81  | 56.45   | 642.77  | 16.69   | 0.63    | 11.2    |
  | S3     | 110.47  | 0.04    | 38.89   | 1397.95 | 2361.45 | 107.62  | 1598.5  | 76.41   | 1382.84 |
  | S4     | 1458.85 | 1049.27 | 597.32  | 1731.9  | 69.09   | 1227.17 | 1187.55 | 1017.16 | 52.15   |
  | S5     | 0.38    | 2315.52 | 1313.06 | 1253.71 | 50.24   | 29.19   | 60.17   | 1077.35 | 70.11   |
  | S6     | 58.2    | 1395.81 | 84.6    | 830.64  | 1003.86 | 631.17  | 31.13   | 1.4     | 246.24  |
  | S7     | 1255.23 | 1382.31 | 78.79   | 829.02  | 67.31   | 877.35  | 185.28  | 221.98  | 0.05    |
  | S8     | 1990.09 | 1.23    | 38.97   | 1396.35 | 112.54  | 107.54  | 1596.74 | 76.32   | 1183.79 |

- $d_j$: Demand of dealership $j$.

  - $d_{\text{C1}} = 4,\!742,\!532,\!000$
  - $d_{\text{C2}} = 1,\!600,\!594,\!000$
  - $d_{\text{C3}} = 5,\!086,\!889,\!000$
  - $d_{\text{C4}} = 1,\!027,\!326,\!000$
  - $d_{\text{C5}} = 11,\!926,\!044,\!000$
  - $d_{\text{C6}} = 9,\!058,\!407,\!000$
  - $d_{\text{C7}} = 5,\!344,\!367,\!000$
  - $d_{\text{C8}} = 677,\!201,\!000$
  - $d_{\text{C9}} = 3,\!236,\!493,\!000$

#### Decision Variables

- $y_i \in \{0,1\}$: 1 if supplier $i$ is open, 0 otherwise.
- $x_{ij} \geq 0$: Number of vehicles supplied from supplier $i$ to dealership $j$ (continuous, as the question describes $x_{ij}$ as a proportion/quantity).

#### Objective Function

Minimize total cost (fixed + transportation):

$$
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

#### Constraints

1. **Demand Satisfaction:** Each dealership's demand must be fully met.

   $$
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   $$

2. **Supplier Activation:** Vehicles can only be supplied from an open supplier.

   $$
   x_{ij} \leq d_j y_i, \quad \forall i \in I, \forall j \in J
   $$

   (This ensures $x_{ij} = 0$ if $y_i = 0$, and allows up to $d_j$ if $y_i = 1$.)

3. **Variable Domains:**

   $$
   y_i \in \{0,1\}, \quad \forall i \in I
   $$
   $$
   x_{ij} \geq 0, \quad \forall i \in I, \forall j \in J
   $$

---

#### Complete Model (Numerical Formulation)

Minimize:
$$
100.64\,y_{\text{S1}} + 98.72\,y_{\text{S2}} + 100.18\,y_{\text{S3}} + 96.58\,y_{\text{S4}} + 95.75\,y_{\text{S5}} + 99.06\,y_{\text{S6}} + 101.78\,y_{\text{S7}} + 93.86\,y_{\text{S8}}
$$
$$
+ \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$
where $c_{ij}$ is as in the table above.

Subject to, for each $j$:

- $\sum_{i \in I} x_{ij} = d_j$

For each $i, j$:

- $x_{ij} \leq d_j y_i$

Variable domains:

- $y_i \in \{0,1\}$ for all $i$
- $x_{ij} \geq 0$ for all $i, j$

---

#### Data Used

- **Suppliers:** S1, S2, S3, S4, S5, S6, S7, S8
- **Dealerships:** C1, C2, C3, C4, C5, C6, C7, C8, C9
- **Fixed Costs:** as above
- **Transportation Costs:** as above
- **Demands:** as above

All data and identifiers are preserved in source order.