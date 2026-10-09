**Retrieved Data**

---

### 1. PotentialWarehouses_Costs.csv  
**(Preserve: Warehouse ID, Opening Cost, Capacity, Source Row Position)**

| Source Row | Warehouse (i) | Opening Cost (fi) | Capacity (units) |
|------------|---------------|-------------------|------------------|
| 1          | 1             | 3000              | 180              |
| 2          | 2             | 3200              | 160              |
| 3          | 3             | 3100              | 200              |
| 4          | 4             | 2800              | 150              |
| 5          | 5             | 3500              | 170              |
| 6          | 6             | 2700              | 190              |
| 7          | 7             | 2900              | 160              |
| 8          | 8             | 3050              | 175              |
| 9          | 9             | 3100              | 170              |
| 10         | 10            | 2200              | 180              |
| 11         | 11            | 2890              | 190              |

---

### 2. Stores_Demands.csv  
**(Preserve: Store ID, Demand, Source Row Position)**

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

### 3. TransportationCost.csv  
**(Preserve: Matrix orientation, Warehouse and Store IDs, Cost values, Source Row Position)**

- **Matrix: c_ij = cost from warehouse i to store j**
- **Rows: Warehouse (i) = 1 to 11**
- **Columns: Store (j) = 1 to 11**

| Source Row | Warehouse (i) | W1 | W2 | W3 | W4 | W5 | W6 | W7 | W8 | W9 | W10 | W11 |
|------------|---------------|----|----|----|----|----|----|----|----|----|-----|-----|
| 1          | 1             | 12 | 11 | 14 | 15 | 17 | 13 | 12 | 16 | 16 | 14  | 15  |
| 2          | 2             | 17 | 19 | 15 | 20 | 18 | 14 | 17 | 15 | 13 | 15  | 16  |
| 3          | 3             | 13 | 14 | 12 | 14 | 16 | 15 | 11 | 14 | 16 | 18  | 17  |
| 4          | 4             | 18 | 16 | 17 | 13 | 18 | 17 | 14 | 19 | 16 | 13  | 18  |
| 5          | 5             | 10 | 13 | 12 | 19 | 15 | 11 | 12 | 14 | 12 | 15  | 17  |
| 6          | 6             | 15 | 12 | 14 | 16 | 13 | 17 | 16 | 16 | 14 | 18  | 19  |
| 7          | 7             | 14 | 13 | 15 | 17 | 12 | 13 | 14 | 15 | 12 | 16  | 14  |
| 8          | 8             | 19 | 16 | 18 | 20 | 17 | 19 | 16 | 18 | 15 | 15  | 18  |
| 9          | 9             | 17 | 18 | 12 | 14 | 16 | 15 | 14 | 17 | 21 | 15  | 18  |
| 10         | 10            | 14 | 13 | 15 | 17 | 16 | 18 | 14 | 19 | 15 | 17  | 19  |
| 11         | 11            | 15 | 13 | 16 | 17 | 11 | 13 | 14 | 15 | 19 | 21  | 13  |

- **Note:** W1–W11 in the matrix columns correspond to Store (j) = 1–11.

---

**All identifiers, coefficients, and values are preserved as in the original sources. No transposition, truncation, or inference has been performed.**