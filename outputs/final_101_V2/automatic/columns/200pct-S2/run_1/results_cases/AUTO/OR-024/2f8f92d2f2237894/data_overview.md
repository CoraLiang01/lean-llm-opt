Here is the complete retrieval of all relevant data from 'demand.csv', 'fixed_cost.csv', and 'transportation_costs.csv', preserving all identifiers and values as required:

---

### demand.csv

| Customer | Demand |
|----------|--------|
| C1       | 1083   |
| C2       | 776    |
| C3       | 16214  |

---

### fixed_cost.csv

| Facility ID | FixedCost |
|-------------|-----------|
| S1          | 102.33    |
| S2          | 94.92     |
| S3          | 91.83     |

---

### transportation_costs.csv

| Facility ID | Customer | Cost per unit |
|-------------|----------|---------------|
| S1          | C1       | 1506.22       |
| S1          | C2       | 70.9          |
| S1          | C3       | 8.44          |
| S2          | C1       | 1732.65       |
| S2          | C2       | 1780.72       |
| S2          | C3       | 567.44        |
| S3          | C1       | 115.66        |
| S3          | C2       | 100.76        |
| S3          | C3       | 64.68         |

---

#### Source-row positions and axis orientation:

- **Facilities (rows):** S1, S2, S3
- **Customers (columns):** C1, C2, C3
- **FixedCost and Capacity** are associated with each Facility ID (S1, S2, S3).
- **Demand** is associated with each Customer ID (C1, C2, C3).
- **Transportation cost matrix** is 3 (facilities) × 3 (customers), with explicit mapping as above.

---

**No capacity data is present; capacity is unresolved evidence, not zero.**

**No extra product/commodity axis is present or inferred.**

**No data has been transposed, truncated, padded, or zero-filled.**

---

This is the complete and unaltered data required for modeling the two-dimensional shipment decision as described.