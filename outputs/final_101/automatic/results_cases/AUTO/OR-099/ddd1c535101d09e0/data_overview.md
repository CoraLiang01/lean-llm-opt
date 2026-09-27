**PotentialWarehouses_Costs.csv**

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

**Stores_Demands.csv**

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

**TransportationCost.csv**  
*Rows: Warehouse (i) W1–W11; Columns: Store (j) W1–W11 (corresponding to Store 1–11)*

| Source Row | To Store 1 | To Store 2 | To Store 3 | To Store 4 | To Store 5 | To Store 6 | To Store 7 | To Store 8 | To Store 9 | To Store 10 | To Store 11 |
|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|-------------|-------------|
| W1         | 12         | 11         | 14         | 15         | 17         | 13         | 12         | 16         | 16         | 14          | 15          |
| W2         | 17         | 19         | 15         | 20         | 18         | 14         | 17         | 15         | 13         | 15          | 16          |
| W3         | 13         | 14         | 12         | 14         | 16         | 15         | 11         | 14         | 16         | 18          | 17          |
| W4         | 18         | 16         | 17         | 13         | 18         | 17         | 14         | 19         | 16         | 13          | 18          |
| W5         | 10         | 13         | 12         | 19         | 15         | 11         | 12         | 14         | 12         | 15          | 17          |
| W6         | 15         | 12         | 14         | 16         | 13         | 17         | 16         | 16         | 14         | 18          | 19          |
| W7         | 14         | 13         | 15         | 17         | 12         | 13         | 14         | 15         | 12         | 16          | 14          |
| W8         | 19         | 16         | 18         | 20         | 17         | 19         | 16         | 18         | 15         | 15          | 18          |
| W9         | 17         | 18         | 12         | 14         | 16         | 15         | 14         | 17         | 21         | 15          | 18          |
| W10        | 14         | 13         | 15         | 17         | 16         | 18         | 14         | 19         | 15         | 17          | 19          |
| W11        | 15         | 13         | 16         | 17         | 11         | 13         | 14         | 15         | 19         | 21          | 13          |

---

**Preserved Identifiers and Source Orientation:**

- **Facility IDs:** Warehouse (i) = 1–11 (W1–W11)
- **Customer IDs:** Store (j) = 1–11 (W1–W11)
- **FixedCost/Capacity:** Each warehouse’s opening cost and capacity is matched by its Warehouse ID.
- **Demand:** Each store’s demand is matched by its Store ID.
- **Cost Matrix:** Each row is a warehouse (W1–W11), each column is a store (W1–W11), with values as transportation costs from warehouse i to store j. The matrix is not transposed or altered.

**No data has been omitted, transposed, or inferred beyond the original files.**