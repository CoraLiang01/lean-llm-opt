**PotentialWarehouses_Costs.csv**

| Row | Warehouse (i) | Opening Cost (fi) | Capacity (units) | warehouse_roof_material | warehouse_inspection_count_2025_q4 |
|-----|---------------|-------------------|------------------|------------------------|-------------------------------------|
| 1   | 1             | 3000              | 180              | Concrete               | 6                                   |
| 2   | 2             | 3200              | 160              | Concrete               | 4                                   |
| 3   | 3             | 3100              | 200              | Steel                  | 4                                   |
| 4   | 4             | 2800              | 150              | Composite              | 2                                   |
| 5   | 5             | 3500              | 170              | Composite              | 6                                   |
| 6   | 6             | 2700              | 190              | Composite              | 3                                   |
| 7   | 7             | 2900              | 160              | Steel                  | 1                                   |
| 8   | 8             | 3050              | 175              | Steel                  | 2                                   |
| 9   | 9             | 3100              | 170              | Steel                  | 1                                   |
| 10  | 10            | 2200              | 180              | Concrete               | 1                                   |
| 11  | 11            | 2890              | 190              | Steel                  | 3                                   |

---

**Stores_Demands.csv**

| Row | Store (j) | Demand (units, dj) | store_staff_training_hours_2025_q4 |
|-----|-----------|--------------------|------------------------------------|
| 1   | 1         | 30                 | 12                                 |
| 2   | 2         | 40                 | 48                                 |
| 3   | 3         | 20                 | 18                                 |
| 4   | 4         | 35                 | 48                                 |
| 5   | 5         | 20                 | 18                                 |
| 6   | 6         | 25                 | 36                                 |
| 7   | 7         | 45                 | 18                                 |
| 8   | 8         | 38                 | 48                                 |
| 9   | 9         | 32                 | 12                                 |
| 10  | 10        | 41                 | 18                                 |
| 11  | 11        | 44                 | 36                                 |

---

**TransportationCost.csv**

*Each row is for a warehouse (W1–W11), each column is a store (1–11). The value is the transportation cost c_ij from warehouse i to store j. The source orientation is preserved:*

| Warehouse (i) | Store 1 | Store 2 | Store 3 | Store 4 | Store 5 | Store 6 | Store 7 | Store 8 | Store 9 | Store 10 | Store 11 |
|---------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|----------|----------|
| W1            | 12      | 11      | 14      | 15      | 17      | 13      | 12      | 16      | 16      | 14       | 15       |
| W2            | 17      | 19      | 15      | 20      | 18      | 14      | 17      | 15      | 13      | 15       | 16       |
| W3            | 13      | 14      | 12      | 14      | 16      | 15      | 11      | 14      | 16      | 18       | 17       |
| W4            | 18      | 16      | 17      | 13      | 18      | 17      | 14      | 19      | 16      | 13       | 18       |
| W5            | 10      | 13      | 12      | 19      | 15      | 11      | 12      | 14      | 12      | 15       | 17       |
| W6            | 15      | 12      | 14      | 16      | 13      | 17      | 16      | 16      | 14      | 18       | 19       |
| W7            | 14      | 13      | 15      | 17      | 12      | 13      | 14      | 15      | 12      | 16       | 14       |
| W8            | 19      | 16      | 18      | 20      | 17      | 19      | 16      | 18      | 15      | 15       | 18       |
| W9            | 17      | 18      | 12      | 14      | 16      | 15      | 14      | 17      | 21      | 15       | 18       |
| W10           | 14      | 13      | 15      | 17      | 16      | 18      | 14      | 19      | 15      | 17       | 19       |
| W11           | 15      | 13      | 16      | 17      | 11      | 13      | 14      | 15      | 19      | 21       | 13       |

---

**Summary of preserved structure:**

- **Facility IDs:** 1–11 (W1–W11), with opening cost and capacity per warehouse.
- **Customer IDs:** 1–11 (Store 1–11), with demand per store.
- **Cost matrix:** c_ij, where i = warehouse (row), j = store (column), shape 11x11, matching warehouse and store IDs.
- **No transposition, truncation, or inferred axes. All data and identifiers are preserved as in the source.**