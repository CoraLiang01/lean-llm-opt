**Retrieved Data**

---

### PotentialWarehouses_Costs.csv  
(Preserving all facility IDs, opening costs, and capacities)

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
(Preserving all customer IDs and demands)

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
(Preserving all cost-matrix axis, with warehouse i as rows and store j as columns; each entry is c_ij)

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

- **Rows:** Warehouse (i) = 1 to 11
- **Columns:** Store (j) = 1 to 11
- **Entry (i, j):** Transportation cost c_ij from warehouse i to store j

---

**All identifiers, values, and matrix orientations are preserved as in the source.**