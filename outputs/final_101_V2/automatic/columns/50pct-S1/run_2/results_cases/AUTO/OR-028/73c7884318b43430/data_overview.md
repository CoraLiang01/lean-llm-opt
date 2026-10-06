Below is the complete retrieval of all data from the three requested sources, with all identifiers, coefficients, and values preserved. Each facility (warehouse) and customer (store) is kept with its original IDs, costs, capacities, and demand. The transportation cost matrix is shown with explicit warehouse (row) and store (column) IDs, as in the source.

---

### PotentialWarehouses_Costs.csv

| Source Row | Warehouse (i) | Opening Cost (fi) | Capacity (units) | record_display_theme | archive_revision_number |
|------------|---------------|-------------------|------------------|---------------------|------------------------|
| 1          | 1             | 3000              | 180              | Olive               | 1                      |
| 2          | 2             | 3200              | 160              | Olive               | 6                      |
| 3          | 3             | 3100              | 200              | Olive               | 4                      |
| 4          | 4             | 2800              | 150              | Olive               | 1                      |
| 5          | 5             | 3500              | 170              | Azure               | 1                      |
| 6          | 6             | 2700              | 190              | Amber               | 1                      |
| 7          | 7             | 2900              | 160              | Amber               | 2                      |
| 8          | 8             | 3050              | 175              | Amber               | 1                      |
| 9          | 9             | 3100              | 170              | Azure               | 2                      |
| 10         | 10            | 2200              | 180              | Amber               | 5                      |
| 11         | 11            | 2890              | 190              | Olive               | 6                      |

---

### Stores_Demands.csv

| Source Row | Store (j) | Demand (units, dj) | archive_revision_number |
|------------|-----------|--------------------|------------------------|
| 1          | 1         | 30                 | 6                      |
| 2          | 2         | 40                 | 5                      |
| 3          | 3         | 20                 | 1                      |
| 4          | 4         | 35                 | 1                      |
| 5          | 5         | 20                 | 3                      |
| 6          | 6         | 25                 | 6                      |
| 7          | 7         | 45                 | 1                      |
| 8          | 8         | 38                 | 6                      |
| 9          | 9         | 32                 | 2                      |
| 10         | 10        | 41                 | 2                      |
| 11         | 11        | 44                 | 4                      |

---

### TransportationCost.csv

Each row is labeled by warehouse (W1–W11), each column by store (1–11). The value is the transportation cost c_ij from warehouse i to store j.

| Source Row | Warehouse (i) | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 | archive_revision_number |
|------------|---------------|----|----|----|----|----|----|----|----|----|-----|-----|------------------------|
| 1          | W1            | 12 | 11 | 14 | 15 | 17 | 13 | 12 | 16 | 16 | 14  | 15  | 5                      |
| 2          | W2            | 17 | 19 | 15 | 20 | 18 | 14 | 17 | 15 | 13 | 15  | 16  | 5                      |
| 3          | W3            | 13 | 14 | 12 | 14 | 16 | 15 | 11 | 14 | 16 | 18  | 17  | 1                      |
| 4          | W4            | 18 | 16 | 17 | 13 | 18 | 17 | 14 | 19 | 16 | 13  | 18  | 4                      |
| 5          | W5            | 10 | 13 | 12 | 19 | 15 | 11 | 12 | 14 | 12 | 15  | 17  | 4                      |
| 6          | W6            | 15 | 12 | 14 | 16 | 13 | 17 | 16 | 16 | 14 | 18  | 19  | 1                      |
| 7          | W7            | 14 | 13 | 15 | 17 | 12 | 13 | 14 | 15 | 12 | 16  | 14  | 6                      |
| 8          | W8            | 19 | 16 | 18 | 20 | 17 | 19 | 16 | 18 | 15 | 15  | 18  | 4                      |
| 9          | W9            | 17 | 18 | 12 | 14 | 16 | 15 | 14 | 17 | 21 | 15  | 18  | 1                      |
| 10         | W10           | 14 | 13 | 15 | 17 | 16 | 18 | 14 | 19 | 15 | 17  | 19  | 3                      |
| 11         | W11           | 15 | 13 | 16 | 17 | 11 | 13 | 14 | 15 | 19 | 21  | 13  | 5                      |

- Warehouse (i): 1–11 (W1–W11)
- Store (j): 1–11 (S1–S11)
- c_ij: Transportation cost from warehouse i to store j

---

**All data is preserved as in the original sources, with all identifiers, coefficients, and values.**