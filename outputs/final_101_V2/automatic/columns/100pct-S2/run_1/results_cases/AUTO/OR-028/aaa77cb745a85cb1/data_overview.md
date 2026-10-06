Below is the complete retrieval of all relevant data from the three files, preserving all facility and customer IDs, costs, capacities, demands, and the full transportation cost matrix. Each value is shown with its source-row position and original orientation.

---

### PotentialWarehouses_Costs.csv

| Warehouse (i) | Opening Cost (fi) | Capacity (units) | Roof Material | warehouse_inspection_count_2025_q4 | warehouse_training_hours_2025_q4 |
|---------------|-------------------|------------------|--------------|------------------------------------|----------------------------------|
| 1             | 3000              | 180              | Concrete     | 6                                  | 18                               |
| 2             | 3200              | 160              | Concrete     | 4                                  | 24                               |
| 3             | 3100              | 200              | Steel        | 4                                  | 18                               |
| 4             | 2800              | 150              | Composite    | 2                                  | 18                               |
| 5             | 3500              | 170              | Composite    | 6                                  | 36                               |
| 6             | 2700              | 190              | Composite    | 3                                  | 24                               |
| 7             | 2900              | 160              | Steel        | 1                                  | 18                               |
| 8             | 3050              | 175              | Steel        | 2                                  | 12                               |
| 9             | 3100              | 170              | Steel        | 1                                  | 24                               |
| 10            | 2200              | 180              | Concrete     | 1                                  | 18                               |
| 11            | 2890              | 190              | Steel        | 3                                  | 24                               |

---

### Stores_Demands.csv

| Store (j) | Demand (units, dj) | store_staff_training_hours_2025_q4 | store_display_window_count |
|-----------|--------------------|------------------------------------|---------------------------|
| 1         | 30                 | 12                                 | 3                         |
| 2         | 40                 | 48                                 | 1                         |
| 3         | 20                 | 18                                 | 5                         |
| 4         | 35                 | 48                                 | 1                         |
| 5         | 20                 | 18                                 | 3                         |
| 6         | 25                 | 36                                 | 1                         |
| 7         | 45                 | 18                                 | 4                         |
| 8         | 38                 | 48                                 | 1                         |
| 9         | 32                 | 12                                 | 3                         |
| 10        | 41                 | 18                                 | 3                         |
| 11        | 44                 | 36                                 | 3                         |

---

### TransportationCost.csv

**Transportation cost c_ij from warehouse i (rows) to store j (columns):**

| Warehouse\Store | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  | 10 | 11 |
|-----------------|----|----|----|----|----|----|----|----|----|----|----|
| 1               | 12 | 11 | 14 | 15 | 17 | 13 | 12 | 16 | 16 | 14 | 15 |
| 2               | 17 | 19 | 15 | 20 | 18 | 14 | 17 | 15 | 13 | 15 | 16 |
| 3               | 13 | 14 | 12 | 14 | 16 | 15 | 11 | 14 | 16 | 18 | 17 |
| 4               | 18 | 16 | 17 | 13 | 18 | 17 | 14 | 19 | 16 | 13 | 18 |
| 5               | 10 | 13 | 12 | 19 | 15 | 11 | 12 | 14 | 12 | 15 | 17 |
| 6               | 15 | 12 | 14 | 16 | 13 | 17 | 16 | 16 | 14 | 18 | 19 |
| 7               | 14 | 13 | 15 | 17 | 12 | 13 | 14 | 15 | 12 | 16 | 14 |
| 8               | 19 | 16 | 18 | 20 | 17 | 19 | 16 | 18 | 15 | 15 | 18 |
| 9               | 17 | 18 | 12 | 14 | 16 | 15 | 14 | 17 | 21 | 15 | 18 |
| 10              | 14 | 13 | 15 | 17 | 16 | 18 | 14 | 19 | 15 | 17 | 19 |
| 11              | 15 | 13 | 16 | 17 | 11 | 13 | 14 | 15 | 19 | 21 | 13 |

- **Rows:** Warehouses 1–11 (facility IDs)
- **Columns:** Stores 1–11 (customer IDs)
- **Each cell:** c_ij = transportation cost from warehouse i to store j

---

**All data is preserved in original orientation and shape, with all identifiers and values intact.**