Mathematical Model for Colorado Motor Vehicle Sales Facility Location and Transportation Problem

Sets:
- Let I = {S1, S2, S3, S4, S5, S6, S7, S8} be the set of suppliers.
- Let J = {C1, C2, C3, C4, C5, C6, C7, C8, C9} be the set of dealerships.

Parameters:
- Fixed cost for opening supplier i ∈ I: 
  - f_S1 = 100.64
  - f_S2 = 98.72
  - f_S3 = 100.18
  - f_S4 = 96.58
  - f_S5 = 95.75
  - f_S6 = 99.06
  - f_S7 = 101.78
  - f_S8 = 93.86

- Demand at dealership j ∈ J:
  - d_C1 = 4,742,532,000
  - d_C2 = 1,600,594,000
  - d_C3 = 5,086,889,000
  - d_C4 = 1,027,326,000
  - d_C5 = 11,926,044,000
  - d_C6 = 9,058,407,000
  - d_C7 = 5,344,367,000
  - d_C8 = 677,201,000
  - d_C9 = 3,236,493,000

- Transportation cost per vehicle from supplier i to dealership j: c_{ij} (matrix below)

|        | C1      | C2      | C3      | C4      | C5      | C6      | C7      | C8      | C9      |
|--------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| S1     | 1091.04 | 85.72   | 99.08   | 747.35  | 893.86  | 23.65   | 15.11   | 15.03   | 497.88  |
| S2     | 58.88   | 1617.16 | 1786.44 | 951.81  | 56.45   | 642.77  | 16.69   | 0.63    | 11.2    |
| S3     | 110.47  | 0.04    | 38.89   | 1397.95 | 2361.45 | 107.62  | 1598.5  | 76.41   | 1382.84 |
| S4     | 1458.85 | 1049.27 | 597.32  | 1731.9  | 69.09   | 1227.17 | 1187.55 | 1017.16 | 52.15   |
| S5     | 0.38    | 2315.52 | 1313.06 | 1253.71 | 50.24   | 29.19   | 60.17   | 1077.35 | 70.11   |
| S6     | 58.2    | 1395.81 | 84.6    | 830.64  | 1003.86 | 631.17  | 31.13   | 1.4     | 246.24  |
| S7     | 1255.23 | 1382.31 | 78.79   | 829.02  | 67.31   | 877.35  | 185.28  | 221.98  | 0.05    |
| S8     | 1990.09 | 1.23    | 38.97   | 1396.35 | 112.54  | 107.54  | 1596.74 | 76.32   | 1183.79 |

Decision Variables:
- y_i ∈ {0,1} for each i ∈ I: 1 if supplier i is open, 0 otherwise.
- x_{ij} ≥ 0 for each i ∈ I, j ∈ J: number of vehicles supplied from supplier i to dealership j.

Objective:
Minimize the total cost, which is the sum of fixed costs for opening suppliers and the total transportation cost:
\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]
where f_i and c_{ij} are as specified above.

Constraints:
1. Demand satisfaction at each dealership:
   \[
   \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J
   \]
   (Each dealership's demand must be fully met.)

2. Supply only from open suppliers:
   \[
   x_{ij} \leq d_j y_i \quad \forall i \in I, \forall j \in J
   \]
   (A supplier can only supply to a dealership if it is open; the maximum it could supply to any dealership is that dealership's full demand.)

3. Binary and non-negativity:
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]
   \[
   x_{ij} \geq 0 \quad \forall i \in I, j \in J
   \]

Full Model Summary:
- Sets, parameters, and all data (fixed costs, demands, transportation costs) are as above.
- Decision variables: y_i (binary), x_{ij} (continuous, ≥0).
- Objective: Minimize total fixed + transportation cost.
- Constraints: Meet all dealership demands, only supply from open suppliers, variable domains.

This model fully preserves the original objective sense, all expressions, and constant terms as provided in the CSV data.