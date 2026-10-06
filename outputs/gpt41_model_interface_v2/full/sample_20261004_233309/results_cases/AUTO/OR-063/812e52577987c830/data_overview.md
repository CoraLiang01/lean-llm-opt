Below is the complete retrieval of all data from the three files as requested, with all identifiers and values preserved. The data is organized by source and retains the original axis orientation and shape.

---

### 1. Demand Data (`demand.csv`)
| Customer | Demand |
|----------|--------|
| C1       | 1083   |
| C2       | 776    |
| C3       | 16214  |
| C4       | 553    |
| C5       | 17106  |
| C6       | 594    |
| C7       | 732    |

---

### 2. Fixed Cost Data (`fixed_cost.csv`)
| Facility (Warehouse) | Fixed Cost |
|----------------------|------------|
| S1                   | 102.33     |
| S2                   | 94.92      |
| S3                   | 91.83      |
| S4                   | 98.71      |
| S5                   | 95.73      |
| S6                   | 99.96      |
| S7                   | 98.16      |

---

### 3. Transportation Cost Matrix (`transportation_costs.csv`)
Each row corresponds to a warehouse (S1–S7), and each column to a customer (C1–C7). The value is the per-unit transportation cost from warehouse Si to customer Cj.

| Warehouse\Customer | C1      | C2      | C3      | C4      | C5      | C6      | C7      |
|--------------------|---------|---------|---------|---------|---------|---------|---------|
| S1                 | 1506.22 | 70.90   | 8.44    | 260.27  | 197.47  | 71.71   | 61.19   |
| S2                 | 1732.65 | 1780.72 | 567.44  | 448.68  | 29.00   | 1484.91 | 963.92  |
| S3                 | 115.66  | 100.76  | 64.68   | 1324.53 | 64.99   | 134.88  | 2102.83 |
| S4                 | 1254.78 | 1115.63 | 52.31   | 1036.16 | 892.63  | 1464.04 | 1383.41 |
| S5                 | 42.90   | 891.01  | 1013.94 | 1128.72 | 58.91   | 42.89   | 1570.31 |
| S6                 | 0.70    | 139.46  | 70.03   | 79.15   | 1482.00 | 0.91    | 110.46  |
| S7                 | 1732.30 | 1780.44 | 486.50  | 523.74  | 522.08  | 82.48   | 826.41  |

---

**All facility IDs, customer IDs, fixed costs, and demand values are preserved. The cost matrix is presented with explicit warehouse (facility) and customer axes, matching the original data orientation. No capacity data is present in the provided context.**