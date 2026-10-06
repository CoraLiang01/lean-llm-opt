Below is the complete retrieval of all data from the three specified sources, with all identifiers, coefficients, and values preserved. The data is organized by source and retains all original facility and customer IDs, costs, capacities, demands, and the full cost matrix with explicit axis orientation.

---

### PotentialWarehouses_Costs.csv

| Warehouse (i) | Opening Cost (fi) | Capacity (units) |
|---------------|-------------------|------------------|
| 1             | 3000              | 180              |
| 2             | 3200              | 160              |
| 3             | 3100              | 200              |
| 4             | 2800              | 150              |
| 5             | 3500              | 170              |
| 6             | 2700              | 190              |
| 7             | 2900              | 160              |
| 8             | 3050              | 175              |
| 9             | 3100              | 170              |
| 10            | 2200              | 180              |
| 11            | 2890              | 190              |

---

### Stores_Demands.csv

| Store (j) | Demand (units, dj) |
|-----------|--------------------|
| 1         | 30                 |
| 2         | 40                 |
| 3         | 20                 |
| 4         | 35                 |
| 5         | 20                 |
| 6         | 25                 |
| 7         | 45                 |
| 8         | 38                 |
| 9         | 32                 |
| 10        | 41                 |
| 11        | 44                 |

---

### TransportationCost.csv

#### Cost from Warehouse i (rows) to Store j (columns)
- **Rows:** W1 to W11 (Warehouse 1 to 11)
- **Columns:** W1 to W11 (Store 1 to 11)

| Source Row (Warehouse) | Store 1 | Store 2 | Store 3 | Store 4 | Store 5 | Store 6 | Store 7 | Store 8 | Store 9 | Store 10 | Store 11 |
|-----------------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|----------|----------|
| W1 (Warehouse 1)      | 12      | 11      | 14      | 15      | 17      | 13      | 12      | 16      | 16      | 14       | 15       |
| W2 (Warehouse 2)      | 17      | 19      | 15      | 20      | 18      | 14      | 17      | 15      | 13      | 15       | 16       |
| W3 (Warehouse 3)      | 13      | 14      | 12      | 14      | 16      | 15      | 11      | 14      | 16      | 18       | 17       |
| W4 (Warehouse 4)      | 18      | 16      | 17      | 13      | 18      | 17      | 14      | 19      | 16      | 13       | 18       |
| W5 (Warehouse 5)      | 10      | 13      | 12      | 19      | 15      | 11      | 12      | 14      | 12      | 15       | 17       |
| W6 (Warehouse 6)      | 15      | 12      | 14      | 16      | 13      | 17      | 16      | 16      | 14      | 18       | 19       |
| W7 (Warehouse 7)      | 14      | 13      | 15      | 17      | 12      | 13      | 14      | 15      | 12      | 16       | 14       |
| W8 (Warehouse 8)      | 19      | 16      | 18      | 20      | 17      | 19      | 16      | 18      | 15      | 15       | 18       |
| W9 (Warehouse 9)      | 17      | 18      | 12      | 14      | 16      | 15      | 14      | 17      | 21      | 15       | 18       |
| W10 (Warehouse 10)    | 14      | 13      | 15      | 17      | 16      | 18      | 14      | 19      | 15      | 17       | 19       |
| W11 (Warehouse 11)    | 15      | 13      | 16      | 17      | 11      | 13      | 14      | 15      | 19      | 21       | 13       |

- **Note:** Each row corresponds to a warehouse (W1 = Warehouse 1, etc.), and each column corresponds to a store (W1 = Store 1, etc.). The value at (row i, column j) is the transportation cost from warehouse i to store j.

---

**All data is preserved as per the original files, with explicit mapping of facility and customer IDs, costs, capacities, demands, and the full cost matrix.**