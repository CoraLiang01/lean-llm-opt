**Retrieved Data**

---

### PotentialWarehouses_Costs.csv  
(Preserving all facility IDs, opening costs, capacities, and relevant attributes)

| Source Row | Warehouse (i) | Opening Cost (fi) | Capacity (units) | Roof Material | Inspection Count (2025 Q4) | Training Hours (2025 Q4) |
|------------|---------------|-------------------|------------------|--------------|----------------------------|--------------------------|
| 1          | 1             | 3000              | 180              | Concrete     | 6                          | 18                       |
| 2          | 2             | 3200              | 160              | Concrete     | 4                          | 24                       |
| 3          | 3             | 3100              | 200              | Steel        | 4                          | 18                       |
| 4          | 4             | 2800              | 150              | Composite    | 2                          | 18                       |
| 5          | 5             | 3500              | 170              | Composite    | 6                          | 36                       |
| 6          | 6             | 2700              | 190              | Composite    | 3                          | 24                       |
| 7          | 7             | 2900              | 160              | Steel        | 1                          | 18                       |
| 8          | 8             | 3050              | 175              | Steel        | 2                          | 12                       |
| 9          | 9             | 3100              | 170              | Steel        | 1                          | 24                       |
| 10         | 10            | 2200              | 180              | Concrete     | 1                          | 18                       |
| 11         | 11            | 2890              | 190              | Steel        | 3                          | 24                       |

---

### Stores_Demands.csv  
(Preserving all customer IDs, demands, and relevant attributes)

| Source Row | Store (j) | Demand (units, dj) | Staff Training Hours (2025 Q4) | Display Window Count |
|------------|-----------|--------------------|-------------------------------|---------------------|
| 1          | 1         | 30                 | 12                            | 3                   |
| 2          | 2         | 40                 | 48                            | 1                   |
| 3          | 3         | 20                 | 18                            | 5                   |
| 4          | 4         | 35                 | 48                            | 1                   |
| 5          | 5         | 20                 | 18                            | 3                   |
| 6          | 6         | 25                 | 36                            | 1                   |
| 7          | 7         | 45                 | 18                            | 4                   |
| 8          | 8         | 38                 | 48                            | 1                   |
| 9          | 9         | 32                 | 12                            | 3                   |
| 10         | 10        | 41                 | 18                            | 3                   |
| 11         | 11        | 44                 | 36                            | 3                   |

---

### TransportationCost.csv  
(Preserving the cost matrix: warehouse i to store j, with explicit axis labels and source orientation)

| Source Row | Warehouse (i) | W1 | W2 | W3 | W4 | W5 | W6 | W7 | W8 | W9 | W10 | W11 |
|------------|---------------|----|----|----|----|----|----|----|----|----|-----|-----|
| 1          | W1            | 12 | 11 | 14 | 15 | 17 | 13 | 12 | 16 | 16 | 14  | 15  |
| 2          | W2            | 17 | 19 | 15 | 20 | 18 | 14 | 17 | 15 | 13 | 15  | 16  |
| 3          | W3            | 13 | 14 | 12 | 14 | 16 | 15 | 11 | 14 | 16 | 18  | 17  |
| 4          | W4            | 18 | 16 | 17 | 13 | 18 | 17 | 14 | 19 | 16 | 13  | 18  |
| 5          | W5            | 10 | 13 | 12 | 19 | 15 | 11 | 12 | 14 | 12 | 15  | 17  |
| 6          | W6            | 15 | 12 | 14 | 16 | 13 | 17 | 16 | 16 | 14 | 18  | 19  |
| 7          | W7            | 14 | 13 | 15 | 17 | 12 | 13 | 14 | 15 | 12 | 16  | 14  |
| 8          | W8            | 19 | 16 | 18 | 20 | 17 | 19 | 16 | 18 | 15 | 15  | 18  |
| 9          | W9            | 17 | 18 | 12 | 14 | 16 | 15 | 14 | 17 | 21 | 15  | 18  |
| 10         | W10           | 14 | 13 | 15 | 17 | 16 | 18 | 14 | 19 | 15 | 17  | 19  |
| 11         | W11           | 15 | 13 | 16 | 17 | 11 | 13 | 14 | 15 | 19 | 21  | 13  |

- **Rows:** Warehouses (W1 to W11, corresponding to Warehouse (i) 1 to 11)
- **Columns:** Stores (W1 to W11, corresponding to Store (j) 1 to 11)
- **Values:** Transportation cost c_ij from warehouse i to store j

---

**All data is preserved with explicit facility and customer IDs, fixed costs, capacities, demands, and the full cost matrix, with original source row positions and axis orientation.**