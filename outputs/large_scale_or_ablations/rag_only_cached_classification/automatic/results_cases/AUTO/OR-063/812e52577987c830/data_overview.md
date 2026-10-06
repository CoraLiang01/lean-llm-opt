Here is all the data from the transportation_costs.csv file, preserving the original facility (warehouse) and customer (musician/band) IDs, with the matrix axis and source-row orientation retained:

| Facility (Warehouse) | C1      | C2      | C3      | C4      | C5      | C6      | C7      |
|----------------------|---------|---------|---------|---------|---------|---------|---------|
| S1                   | 1506.22 | 70.90   | 8.44    | 260.27  | 197.47  | 71.71   | 61.19   |
| S2                   | 1732.65 | 1780.72 | 567.44  | 448.68  | 29.00   | 1484.91 | 963.92  |
| S3                   | 115.66  | 100.76  | 64.68   | 1324.53 | 64.99   | 134.88  | 2102.83 |
| S4                   | 1254.78 | 1115.63 | 52.31   | 1036.16 | 892.63  | 1464.04 | 1383.41 |
| S5                   | 42.90   | 891.01  | 1013.94 | 1128.72 | 58.91   | 42.89   | 1570.31 |
| S6                   | 0.70    | 139.46  | 70.03   | 79.15   | 1482.00 | 0.91    | 110.46  |
| S7                   | 1732.30 | 1780.44 | 486.50  | 523.74  | 522.08  | 82.48   | 826.41  |

- Facility (Warehouse) IDs: S1, S2, S3, S4, S5, S6, S7
- Customer (Musician/Band) IDs: C1, C2, C3, C4, C5, C6, C7

Each entry x_{ij} represents the transportation cost per unit from warehouse S_i to customer C_j. The matrix is 7 (warehouses) × 7 (customers), with all axis labels and source-row positions preserved.