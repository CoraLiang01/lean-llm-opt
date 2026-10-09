**Retrieved Data**

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

**Cost matrix c_ij: cost from warehouse i to store j**

| Warehouse\Store | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  | 10 | 11 |
|-----------------|----|----|----|----|----|----|----|----|----|----|----|
| 1               | 12 | 17 | 13 | 18 | 10 | 15 | 14 | 19 | 17 | 14 | 15 |
| 2               | 11 | 19 | 14 | 16 | 13 | 12 | 13 | 16 | 18 | 13 | 13 |
| 3               | 14 | 15 | 12 | 17 | 12 | 14 | 15 | 18 | 12 | 15 | 16 |
| 4               | 15 | 20 | 14 | 13 | 19 | 16 | 17 | 20 | 14 | 17 | 17 |
| 5               | 17 | 18 | 16 | 18 | 15 | 13 | 12 | 17 | 16 | 16 | 11 |
| 6               | 13 | 14 | 15 | 17 | 11 | 17 | 13 | 19 | 15 | 18 | 13 |
| 7               | 12 | 17 | 11 | 14 | 12 | 16 | 14 | 16 | 21 | 15 | 19 |
| 8               | 16 | 15 | 14 | 19 | 14 | 16 | 15 | 18 | 17 | 19 | 15 |
| 9               | 16 | 13 | 16 | 16 | 12 | 14 | 12 | 15 | 21 | 15 | 19 |
| 10              | 14 | 15 | 18 | 13 | 15 | 18 | 16 | 15 | 15 | 17 | 21 |
| 11              | 15 | 16 | 17 | 18 | 17 | 19 | 14 | 18 | 18 | 19 | 13 |

---

**Preserved Identifiers and Source Orientation**

- Facility IDs: 1–11 (warehouses)
- Customer IDs: 1–11 (stores)
- FixedCost and Capacity are matched by warehouse ID.
- Demand is matched by store ID.
- Cost matrix c_ij is indexed as [warehouse i, store j], shape (11 warehouses × 11 stores), with all original values and order preserved.

---

**No data has been transposed, truncated, or inferred beyond the original files.**