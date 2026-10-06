Below is the complete retrieval of all relevant data from PotentialWarehouses_Costs.csv, Stores_Demands.csv, and TransportationCost.csv, preserving all facility and customer IDs, FixedCost, Capacity, Demand, and the full cost-matrix with explicit axis labeling and source-row orientation.

---

### PotentialWarehouses_Costs.csv

| Warehouse (i) | archive_revision_number | document_page_count | Opening Cost (fi) | record_display_theme | Capacity (units) |
|---------------|------------------------|--------------------|-------------------|---------------------|------------------|
| 1             | 1                      | 6                  | 3000              | Olive               | 180              |
| 2             | 6                      | 6                  | 3200              | Olive               | 160              |
| 3             | 4                      | 16                 | 3100              | Olive               | 200              |
| 4             | 1                      | 8                  | 2800              | Olive               | 150              |
| 5             | 1                      | 6                  | 3500              | Azure               | 170              |
| 6             | 1                      | 2                  | 2700              | Amber               | 190              |
| 7             | 2                      | 6                  | 2900              | Amber               | 160              |
| 8             | 1                      | 16                 | 3050              | Amber               | 175              |
| 9             | 2                      | 4                  | 3100              | Azure               | 170              |
| 10            | 5                      | 6                  | 2200              | Amber               | 180              |
| 11            | 6                      | 12                 | 2890              | Olive               | 190              |

---

### Stores_Demands.csv

| Store (j) | archive_revision_number | Demand (units, dj) | document_page_count |
|-----------|------------------------|--------------------|---------------------|
| 1         | 6                      | 30                 | 8                   |
| 2         | 5                      | 40                 | 12                  |
| 3         | 1                      | 20                 | 12                  |
| 4         | 1                      | 35                 | 4                   |
| 5         | 3                      | 20                 | 4                   |
| 6         | 6                      | 25                 | 16                  |
| 7         | 1                      | 45                 | 12                  |
| 8         | 6                      | 38                 | 8                   |
| 9         | 2                      | 32                 | 4                   |
| 10        | 2                      | 41                 | 16                  |
| 11        | 4                      | 44                 | 16                  |

---

### TransportationCost.csv

#### Cost matrix c_ij: (Rows = Warehouses W1–W11, Columns = Stores 1–11)
Each entry is the transportation cost from warehouse i to store j.

| Warehouse\Store | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  | 10 | 11 |
|-----------------|----|----|----|----|----|----|----|----|----|----|----|
| W1              | 12 | 17 | 13 | 18 | 10 | 15 | 14 | 19 | 17 | 14 | 15 |
| W2              | 11 | 19 | 14 | 16 | 13 | 12 | 13 | 16 | 18 | 13 | 13 |
| W3              | 14 | 15 | 12 | 17 | 12 | 14 | 15 | 18 | 12 | 15 | 16 |
| W4              | 15 | 20 | 14 | 13 | 19 | 16 | 17 | 20 | 14 | 17 | 17 |
| W5              | 17 | 18 | 16 | 18 | 15 | 13 | 12 | 17 | 16 | 16 | 11 |
| W6              | 13 | 14 | 15 | 17 | 11 | 17 | 13 | 19 | 15 | 18 | 13 |
| W7              | 12 | 17 | 11 | 14 | 12 | 16 | 14 | 16 | 14 | 14 | 14 |
| W8              | 16 | 15 | 14 | 19 | 14 | 16 | 15 | 18 | 17 | 19 | 15 |
| W9              | 16 | 13 | 16 | 16 | 12 | 14 | 12 | 15 | 21 | 15 | 19 |
| W10             | 14 | 15 | 18 | 13 | 15 | 18 | 16 | 15 | 15 | 17 | 21 |
| W11             | 15 | 16 | 17 | 18 | 17 | 19 | 14 | 18 | 18 | 19 | 13 |

- Source orientation: Rows = Warehouses (W1–W11), Columns = Stores (1–11)
- All values are preserved as in the source.

---

**All identifiers, values, and matrix axes are preserved as in the original data. No transposition, truncation, or inference has been performed.**