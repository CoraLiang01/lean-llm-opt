**Retrieved Data**

---

### PotentialWarehouses_Costs.csv  
(Preserving all facility IDs, opening costs, and capacities; source-row order)

| Row | Warehouse (i) | Opening Cost (fi) | Capacity (units) |
|-----|---------------|-------------------|------------------|
| 1   | 1             | 3000              | 180              |
| 2   | 2             | 3200              | 160              |
| 3   | 3             | 3100              | 200              |
| 4   | 4             | 2800              | 150              |
| 5   | 5             | 3500              | 170              |
| 6   | 6             | 2700              | 190              |
| 7   | 7             | 2900              | 160              |
| 8   | 8             | 3050              | 175              |
| 9   | 9             | 3100              | 170              |
| 10  | 10            | 2200              | 180              |
| 11  | 11            | 2890              | 190              |

---

### Stores_Demands.csv  
(Preserving all customer IDs and demands; source-row order)

| Row | Store (j) | Demand (units, dj) |
|-----|-----------|--------------------|
| 1   | 1         | 30                 |
| 2   | 2         | 40                 |
| 3   | 3         | 20                 |
| 4   | 4         | 35                 |
| 5   | 5         | 20                 |
| 6   | 6         | 25                 |
| 7   | 7         | 45                 |
| 8   | 8         | 38                 |
| 9   | 9         | 32                 |
| 10  | 10        | 41                 |
| 11  | 11        | 44                 |

---

### TransportationCost.csv  
(Preserving all cost-matrix values, with explicit warehouse and store IDs; source orientation and shape retained)

| Row | Unnamed: 1 | W1 | W2 | W3 | W4 | W5 | W6 | W7 | W8 | W9 | W10 | W11 |
|-----|------------|----|----|----|----|----|----|----|----|----|-----|-----|
| 1   | W1         | 12 | 11 | 14 | 15 | 17 | 13 | 12 | 16 | 16 | 14  | 15  |
| 2   | W2         | 17 | 19 | 15 | 20 | 18 | 14 | 17 | 15 | 13 | 15  | 16  |
| 3   | W3         | 13 | 14 | 12 | 14 | 16 | 15 | 11 | 14 | 16 | 18  | 17  |
| 4   | W4         | 18 | 16 | 17 | 13 | 18 | 17 | 14 | 19 | 16 | 13  | 18  |
| 5   | W5         | 10 | 13 | 12 | 19 | 15 | 11 | 12 | 14 | 12 | 15  | 17  |
| 6   | W6         | 15 | 12 | 14 | 16 | 13 | 17 | 16 | 16 | 14 | 18  | 19  |
| 7   | W7         | 14 | 13 | 15 | 17 | 12 | 13 | 14 | 15 | 12 | 16  | 14  |
| 8   | W8         | 19 | 16 | 18 | 20 | 17 | 19 | 16 | 18 | 15 | 15  | 18  |
| 9   | W9         | 17 | 18 | 12 | 14 | 16 | 15 | 14 | 17 | 21 | 15  | 18  |
| 10  | W10        | 14 | 13 | 15 | 17 | 16 | 18 | 14 | 19 | 15 | 17  | 19  |
| 11  | W11        | 15 | 13 | 16 | 17 | 11 | 13 | 14 | 15 | 19 | 21  | 13  |

- **Rows**: Warehouses W1–W11 (corresponding to Warehouse (i) 1–11)
- **Columns**: Stores W1–W11 (corresponding to Store (j) 1–11)
- **Cell (i, j)**: Transportation cost from warehouse i to store j

---

**All identifiers, values, and matrix shapes are preserved as in the source.**