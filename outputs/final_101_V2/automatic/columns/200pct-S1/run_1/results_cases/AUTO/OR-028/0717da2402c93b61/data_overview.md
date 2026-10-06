Below is the complete retrieval of all data from the three requested sources, with all identifiers, values, and matrix axes preserved. Each facility (warehouse) and customer (store) retains its original ID, FixedCost, Capacity, Demand, and the full cost matrix c_ij (transportation cost from warehouse i to store j), with explicit row and column IDs and original orientation.

---

### PotentialWarehouses_Costs.csv

| Source Row | Warehouse (i) | archive_revision_number | document_page_count | Opening Cost (fi) | archive_storage_medium | record_display_theme | document_template_family | record_view_count | Capacity (units) |
|------------|---------------|------------------------|--------------------|-------------------|-----------------------|---------------------|-------------------------|-------------------|------------------|
| 1          | 1             | 1                      | 6                  | 3000              | Digital               | Olive               | Landscape               | 76                | 180              |
| 2          | 2             | 6                      | 6                  | 3200              | Hybrid                | Olive               | Compact                 | 76                | 160              |
| 3          | 3             | 4                      | 16                 | 3100              | Digital               | Olive               | Compact                 | 76                | 200              |
| 4          | 4             | 1                      | 8                  | 2800              | Digital               | Olive               | Landscape               | 76                | 150              |
| 5          | 5             | 1                      | 6                  | 3500              | Hybrid                | Azure               | Landscape               | 12                | 170              |
| 6          | 6             | 1                      | 2                  | 2700              | Paper                 | Amber               | Standard                | 43                | 190              |
| 7          | 7             | 2                      | 6                  | 2900              | Hybrid                | Amber               | Standard                | 12                | 160              |
| 8          | 8             | 1                      | 16                 | 3050              | Hybrid                | Amber               | Standard                | 58                | 175              |
| 9          | 9             | 2                      | 4                  | 3100              | Digital               | Azure               | Landscape               | 58                | 170              |
| 10         | 10            | 5                      | 6                  | 2200              | Digital               | Amber               | Compact                 | 76                | 180              |
| 11         | 11            | 6                      | 12                 | 2890              | Paper                 | Olive               | Standard                | 76                | 190              |

---

### Stores_Demands.csv

| Source Row | Store (j) | archive_revision_number | record_view_count | archive_batch_number | Demand (units, dj) | document_page_count |
|------------|-----------|------------------------|-------------------|---------------------|--------------------|--------------------|
| 1          | 1         | 6                      | 76                | 305                 | 30                 | 8                  |
| 2          | 2         | 5                      | 27                | 301                 | 40                 | 12                 |
| 3          | 3         | 1                      | 27                | 304                 | 20                 | 12                 |
| 4          | 4         | 1                      | 58                | 305                 | 35                 | 4                  |
| 5          | 5         | 3                      | 43                | 304                 | 20                 | 4                  |
| 6          | 6         | 6                      | 76                | 304                 | 25                 | 16                 |
| 7          | 7         | 1                      | 58                | 304                 | 45                 | 12                 |
| 8          | 8         | 6                      | 91                | 301                 | 38                 | 8                  |
| 9          | 9         | 2                      | 27                | 301                 | 32                 | 4                  |
| 10         | 10        | 2                      | 43                | 302                 | 41                 | 16                 |
| 11         | 11        | 4                      | 27                | 302                 | 44                 | 16                 |

---

### TransportationCost.csv

Each row below corresponds to a warehouse (i), each column to a store (j). The value is the transportation cost c_ij from warehouse i to store j. The original orientation and shape are preserved.

| Source Row | Warehouse (i) | Store 1 | Store 2 | Store 3 | Store 4 | Store 5 | Store 6 | Store 7 | Store 8 | Store 9 | Store 10 | Store 11 |
|------------|---------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|----------|----------|
| 1          | 1             | 12      | 11      | 14      | 15      | 17      | 13      | 12      | 16      | 16      | 14       | 15       |
| 2          | 2             | 17      | 19      | 15      | 20      | 18      | 14      | 17      | 15      | 13      | 15       | 16       |
| 3          | 3             | 13      | 14      | 12      | 14      | 16      | 15      | 11      | 14      | 16      | 18       | 17       |
| 4          | 4             | 18      | 16      | 17      | 13      | 18      | 17      | 14      | 19      | 16      | 13       | 18       |
| 5          | 5             | 10      | 13      | 12      | 19      | 15      | 11      | 12      | 14      | 12      | 15       | 17       |
| 6          | 6             | 15      | 12      | 14      | 16      | 13      | 17      | 16      | 16      | 14      | 18       | 19       |
| 7          | 7             | 14      | 13      | 15      | 17      | 12      | 13      | 14      | 15      | 12      | 16       | 14       |
| 8          | 8             | 19      | 16      | 18      | 20      | 17      | 19      | 16      | 18      | 15      | 15       | 18       |
| 9          | 9             | 17      | 18      | 12      | 14      | 16      | 15      | 14      | 17      | 21      | 15       | 18       |
| 10         | 10            | 14      | 13      | 15      | 17      | 16      | 18      | 14      | 19      | 15      | 17       | 19       |
| 11         | 11            | 15      | 13      | 16      | 17      | 11      | 13      | 14      | 15      | 19      | 21       | 13       |

---

**All data is preserved with original IDs, values, and matrix orientation.**