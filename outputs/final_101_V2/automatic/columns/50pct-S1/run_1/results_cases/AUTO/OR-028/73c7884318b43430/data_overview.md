Below is the complete retrieval of all data from the three specified sources, with all identifiers, coefficients, and values preserved. Each facility (warehouse) and customer (store) retains its original ID, FixedCost, Capacity, Demand, and the full cost matrix (c_ij) with explicit axis orientation and shape.

---

### PotentialWarehouses_Costs.csv

| Source Row | Warehouse (i) | FixedCost (Opening Cost fi) | Capacity (units) | record_display_theme |
|------------|---------------|-----------------------------|------------------|---------------------|
| 1          | 1             | 3000                        | 180              | Olive               |
| 2          | 2             | 3200                        | 160              | Olive               |
| 3          | 3             | 3100                        | 200              | Olive               |
| 4          | 4             | 2800                        | 150              | Olive               |
| 5          | 5             | 3500                        | 170              | Azure               |
| 6          | 6             | 2700                        | 190              | Amber               |
| 7          | 7             | 2900                        | 160              | Amber               |
| 8          | 8             | 3050                        | 175              | Amber               |
| 9          | 9             | 3100                        | 170              | Azure               |
| 10         | 10            | 2200                        | 180              | Amber               |
| 11         | 11            | 2890                        | 190              | Olive               |

---

### Stores_Demands.csv

| Source Row | Store (j) | Demand (units, dj) |
|------------|-----------|--------------------|
| 1          | 1         | 30                 |
| 2          | 2         | 40                 |
| 3          | 3         | 20                 |
| 4          | 4         | 35                 |
| 5          | 5         | 20                 |
| 6          | 6         | 25                 |
| 7          | 7         | 45                 |
| 8          | 8         | 38                 |
| 9          | 9         | 32                 |
| 10         | 10        | 41                 |
| 11         | 11        | 44                 |

---

### TransportationCost.csv

- Matrix orientation: Rows = Warehouses (W1–W11), Columns = Stores (1–11)
- Each entry c_ij is the transportation cost from warehouse i to store j.

| Source Row | Warehouse (i) | Store 1 | Store 2 | Store 3 | Store 4 | Store 5 | Store 6 | Store 7 | Store 8 | Store 9 | Store 10 | Store 11 |
|------------|---------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|----------|----------|
| 1          | W1            | 12      | 11      | 14      | 15      | 17      | 13      | 12      | 16      | 16      | 14       | 15       |
| 2          | W2            | 17      | 19      | 15      | 20      | 18      | 14      | 17      | 15      | 13      | 15       | 16       |
| 3          | W3            | 13      | 14      | 12      | 14      | 16      | 15      | 11      | 14      | 16      | 18       | 17       |
| 4          | W4            | 18      | 16      | 17      | 13      | 18      | 17      | 14      | 19      | 16      | 13       | 18       |
| 5          | W5            | 10      | 13      | 12      | 19      | 15      | 11      | 12      | 14      | 12      | 15       | 17       |
| 6          | W6            | 15      | 12      | 14      | 16      | 13      | 17      | 16      | 16      | 14      | 18       | 19       |
| 7          | W7            | 14      | 13      | 15      | 17      | 12      | 13      | 14      | 15      | 12      | 16       | 14       |
| 8          | W8            | 19      | 16      | 18      | 20      | 17      | 19      | 16      | 18      | 15      | 15       | 18       |
| 9          | W9            | 17      | 18      | 12      | 14      | 16      | 15      | 14      | 17      | 21      | 15       | 18       |
| 10         | W10           | 14      | 13      | 15      | 17      | 16      | 18      | 14      | 19      | 15      | 17       | 19       |
| 11         | W11           | 15      | 13      | 16      | 17      | 11      | 13      | 14      | 15      | 19      | 21       | 13       |

---

#### Notes:
- All facility IDs, customer IDs, FixedCost, Capacity, Demand, and cost-matrix axes are preserved as in the source.
- The cost matrix is 11 (warehouses) × 11 (stores), with explicit warehouse and store IDs.
- No data has been transposed, truncated, padded, or inferred beyond the original query.

This is the complete data required for modeling the two-dimensional warehouse location and shipment assignment problem as described.