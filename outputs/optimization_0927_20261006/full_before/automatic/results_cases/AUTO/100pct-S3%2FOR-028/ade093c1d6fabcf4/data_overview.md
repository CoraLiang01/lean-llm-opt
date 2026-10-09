**PotentialWarehouses_Costs.csv**  
| Row | Warehouse (i) | Opening Cost (fi) | Capacity (units) | previous_period_Opening_Cost | previous_period_Capacity_units | previous_period_opening_status |
|-----|---------------|-------------------|------------------|------------------------------|-------------------------------|-------------------------------|
| 1   | 1             | 3000              | 180              | 3435                         | 207                           | UnderReview                   |
| 2   | 2             | 3200              | 160              | 2801                         | 139                           | UnderReview                   |
| 3   | 3             | 3100              | 200              | 3566                         | 207                           | UnderReview                   |
| 4   | 4             | 2800              | 150              | 2783                         | 138                           | Open                          |
| 5   | 5             | 3500              | 170              | 3184                         | 164                           | Open                          |
| 6   | 6             | 2700              | 190              | 2669                         | 167                           | Open                          |
| 7   | 7             | 2900              | 160              | 2601                         | 168                           | Open                          |
| 8   | 8             | 3050              | 175              | 2750                         | 162                           | Open                          |
| 9   | 9             | 3100              | 170              | 3487                         | 190                           | Reserved                      |
| 10  | 10            | 2200              | 180              | 2579                         | 165                           | Open                          |
| 11  | 11            | 2890              | 190              | 3208                         | 212                           | Open                          |

---

**Stores_Demands.csv**  
| Row | Store (j) | Demand (units, dj) | previous_period_Demand_units | two_periods_ago_Demand_units |
|-----|-----------|--------------------|-----------------------------|-----------------------------|
| 1   | 1         | 30                 | 34                          | 34                          |
| 2   | 2         | 40                 | 45                          | 41                          |
| 3   | 3         | 20                 | 21                          | 21                          |
| 4   | 4         | 35                 | 29                          | 40                          |
| 5   | 5         | 20                 | 23                          | 18                          |
| 6   | 6         | 25                 | 26                          | 27                          |
| 7   | 7         | 45                 | 41                          | 50                          |
| 8   | 8         | 38                 | 43                          | 43                          |
| 9   | 9         | 32                 | 26                          | 34                          |
| 10  | 10        | 41                 | 45                          | 36                          |
| 11  | 11        | 44                 | 49                          | 40                          |

---

**TransportationCost.csv**  
Each row corresponds to a warehouse (W1 to W11), each column to a store (1 to 11).  
Values are the transportation cost c_ij from warehouse i to store j.

| Row | Warehouse (i) | Store 1 | Store 2 | Store 3 | Store 4 | Store 5 | Store 6 | Store 7 | Store 8 | Store 9 | Store 10 | Store 11 |
|-----|---------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|----------|----------|
| 1   | W1            | 12      | 11      | 14      | 15      | 17      | 13      | 12      | 16      | 16      | 14       | 15       |
| 2   | W2            | 17      | 19      | 15      | 20      | 18      | 14      | 17      | 15      | 13      | 15       | 16       |
| 3   | W3            | 13      | 14      | 12      | 14      | 16      | 15      | 11      | 14      | 16      | 18       | 17       |
| 4   | W4            | 18      | 16      | 17      | 13      | 18      | 17      | 14      | 19      | 16      | 13       | 18       |
| 5   | W5            | 10      | 13      | 12      | 19      | 15      | 11      | 12      | 14      | 12      | 15       | 17       |
| 6   | W6            | 15      | 12      | 14      | 16      | 13      | 17      | 16      | 16      | 14      | 18       | 19       |
| 7   | W7            | 14      | 13      | 15      | 17      | 12      | 13      | 14      | 15      | 12      | 16       | 14       |
| 8   | W8            | 19      | 16      | 18      | 20      | 17      | 19      | 16      | 18      | 15      | 15       | 18       |
| 9   | W9            | 17      | 18      | 12      | 14      | 16      | 15      | 14      | 17      | 21      | 15       | 18       |
| 10  | W10           | 14      | 13      | 15      | 17      | 16      | 18      | 14      | 19      | 15      | 17       | 19       |
| 11  | W11           | 15      | 13      | 16      | 17      | 11      | 13      | 14      | 15      | 19      | 21       | 13       |

---

**Preserved Identifiers and Structure:**
- Facility IDs: 1–11 (W1–W11)
- Customer IDs: 1–11 (Store 1–11)
- FixedCost and Capacity are matched by facility ID.
- Demand is matched by customer ID.
- Transportation cost matrix is [Warehouse (row) x Store (column)], shape 11x11, with explicit IDs.

**No data has been transposed, truncated, or inferred beyond the original files.**