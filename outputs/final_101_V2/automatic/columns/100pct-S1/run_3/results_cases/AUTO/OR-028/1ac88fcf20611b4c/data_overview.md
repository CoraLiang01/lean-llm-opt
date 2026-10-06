**PotentialWarehouses_Costs.csv**

| Warehouse (i) | archive_revision_number | document_page_count | Opening Cost (fi) | record_display_theme | Capacity (units) | Source Row |
|---------------|------------------------|--------------------|-------------------|---------------------|------------------|------------|
| 1             | 1                      | 6                  | 3000              | Olive               | 180              | 1          |
| 2             | 6                      | 6                  | 3200              | Olive               | 160              | 2          |
| 3             | 4                      | 16                 | 3100              | Olive               | 200              | 3          |
| 4             | 1                      | 8                  | 2800              | Olive               | 150              | 4          |
| 5             | 1                      | 6                  | 3500              | Azure               | 170              | 5          |
| 6             | 1                      | 2                  | 2700              | Amber               | 190              | 6          |
| 7             | 2                      | 6                  | 2900              | Amber               | 160              | 7          |
| 8             | 1                      | 16                 | 3050              | Amber               | 175              | 8          |
| 9             | 2                      | 4                  | 3100              | Azure               | 170              | 9          |
| 10            | 5                      | 6                  | 2200              | Amber               | 180              | 10         |
| 11            | 6                      | 12                 | 2890              | Olive               | 190              | 11         |

---

**Stores_Demands.csv**

| Store (j) | archive_revision_number | Demand (units, dj) | document_page_count | Source Row |
|-----------|------------------------|--------------------|---------------------|------------|
| 1         | 6                      | 30                 | 8                   | 1          |
| 2         | 5                      | 40                 | 12                  | 2          |
| 3         | 1                      | 20                 | 12                  | 3          |
| 4         | 1                      | 35                 | 4                   | 4          |
| 5         | 3                      | 20                 | 4                   | 5          |
| 6         | 6                      | 25                 | 16                  | 6          |
| 7         | 1                      | 45                 | 12                  | 7          |
| 8         | 6                      | 38                 | 8                   | 8          |
| 9         | 2                      | 32                 | 4                   | 9          |
| 10        | 2                      | 41                 | 16                  | 10         |
| 11        | 4                      | 44                 | 16                  | 11         |

---

**TransportationCost.csv**

- Each row is for a warehouse (W1 to W11), each column is a store (1 to 11). The value is c_ij (cost from warehouse i to store j).
- Source orientation: rows = warehouses (W1 to W11), columns = stores (1 to 11).

| Warehouse (i) | W1 | W2 | W3 | W4 | W5 | W6 | W7 | W8 | W9 | W10 | W11 | Source Row |
|---------------|----|----|----|----|----|----|----|----|----|-----|------|------------|
| W1            | 12 | 11 | 14 | 15 | 17 | 13 | 12 | 16 | 16 | 14  | 15   | 1          |
| W2            | 17 | 19 | 15 | 20 | 18 | 14 | 17 | 15 | 13 | 15  | 16   | 2          |
| W3            | 13 | 14 | 12 | 14 | 16 | 15 | 11 | 14 | 16 | 18  | 17   | 3          |
| W4            | 18 | 16 | 17 | 13 | 18 | 17 | 14 | 19 | 16 | 13  | 18   | 4          |
| W5            | 10 | 13 | 12 | 19 | 15 | 11 | 12 | 14 | 12 | 15  | 17   | 5          |
| W6            | 15 | 12 | 14 | 16 | 13 | 17 | 16 | 16 | 14 | 18  | 19   | 6          |
| W7            | 14 | 13 | 15 | 17 | 12 | 13 | 14 | 15 | 12 | 16  | 14   | 7          |
| W8            | 19 | 16 | 18 | 20 | 17 | 19 | 16 | 18 | 15 | 15  | 18   | 8          |
| W9            | 17 | 18 | 12 | 14 | 16 | 15 | 14 | 17 | 21 | 15  | 18   | 9          |
| W10           | 14 | 13 | 15 | 17 | 16 | 18 | 14 | 19 | 15 | 17  | 19   | 10         |
| W11           | 15 | 13 | 16 | 17 | 11 | 13 | 14 | 15 | 19 | 21  | 13   | 11         |

- Matrix shape: 11 warehouses (rows) × 11 stores (columns).
- Axis: Warehouse i (rows, 1–11), Store j (columns, 1–11).
- All values are preserved as in the source.

---

**Summary of preserved identifiers and values:**

- Facility IDs: 1–11 (Warehouse (i)), with Opening Cost (fi) and Capacity (units) per warehouse.
- Customer IDs: 1–11 (Store (j)), with Demand (units, dj) per store.
- Cost matrix: c_ij, with explicit warehouse-to-store mapping, shape 11×11, no transposition or truncation.
- All source row positions are retained for traceability.