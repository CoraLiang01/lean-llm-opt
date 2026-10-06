**PotentialWarehouses_Costs.csv**

| Warehouse (i) | archive_revision_number | document_page_count | Opening Cost (fi) | record_display_theme | Capacity (units) |
|---------------|-----------------------|--------------------|-------------------|---------------------|------------------|
| 1             | 1                     | 6                  | 3000              | Olive               | 180              |
| 2             | 6                     | 6                  | 3200              | Olive               | 160              |
| 3             | 4                     | 16                 | 3100              | Olive               | 200              |
| 4             | 1                     | 8                  | 2800              | Olive               | 150              |
| 5             | 1                     | 6                  | 3500              | Azure               | 170              |
| 6             | 1                     | 2                  | 2700              | Amber               | 190              |
| 7             | 2                     | 6                  | 2900              | Amber               | 160              |
| 8             | 1                     | 16                 | 3050              | Amber               | 175              |
| 9             | 2                     | 4                  | 3100              | Azure               | 170              |
| 10            | 5                     | 6                  | 2200              | Amber               | 180              |
| 11            | 6                     | 12                 | 2890              | Olive               | 190              |

---

**Stores_Demands.csv**

| Store (j) | archive_revision_number | Demand (units, dj) | document_page_count |
|-----------|-----------------------|--------------------|---------------------|
| 1         | 6                     | 30                 | 8                   |
| 2         | 5                     | 40                 | 12                  |
| 3         | 1                     | 20                 | 12                  |
| 4         | 1                     | 35                 | 4                   |
| 5         | 3                     | 20                 | 4                   |
| 6         | 6                     | 25                 | 16                  |
| 7         | 1                     | 45                 | 12                  |
| 8         | 6                     | 38                 | 8                   |
| 9         | 2                     | 32                 | 4                   |
| 10        | 2                     | 41                 | 16                  |
| 11        | 4                     | 44                 | 16                  |

---

**TransportationCost.csv**  
(Matrix: rows = warehouses W1–W11, columns = stores 1–11; values are c_ij, the transportation cost from warehouse i to store j. Source orientation and shape preserved.)

|         | Store 1 | Store 2 | Store 3 | Store 4 | Store 5 | Store 6 | Store 7 | Store 8 | Store 9 | Store 10 | Store 11 |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|----------|----------|
| W1      | 12      | 11      | 14      | 15      | 17      | 13      | 12      | 16      | 16      | 14       | 15       |
| W2      | 17      | 19      | 15      | 20      | 18      | 14      | 17      | 15      | 13      | 15       | 16       |
| W3      | 13      | 14      | 12      | 14      | 16      | 15      | 11      | 14      | 16      | 18       | 17       |
| W4      | 18      | 16      | 17      | 13      | 18      | 17      | 14      | 19      | 16      | 13       | 18       |
| W5      | 10      | 13      | 12      | 19      | 15      | 11      | 12      | 14      | 12      | 15       | 17       |
| W6      | 15      | 12      | 14      | 16      | 13      | 17      | 16      | 16      | 14      | 18       | 19       |
| W7      | 14      | 13      | 15      | 17      | 12      | 13      | 14      | 15      | 12      | 16       | 14       |
| W8      | 19      | 16      | 18      | 20      | 17      | 19      | 16      | 18      | 15      | 15       | 18       |
| W9      | 17      | 18      | 12      | 14      | 16      | 15      | 14      | 17      | 21      | 15       | 18       |
| W10     | 14      | 13      | 15      | 17      | 16      | 18      | 14      | 19      | 15      | 17       | 19       |
| W11     | 15      | 13      | 16      | 17      | 11      | 13      | 14      | 15      | 19      | 21       | 13       |

- All identifiers (warehouse and store IDs) and values are preserved as in the source.
- Warehouse opening costs and capacities are matched by warehouse ID.
- Store demands are matched by store ID.
- Transportation cost matrix is oriented as (warehouse, store) with explicit IDs. No transposition or inference applied.