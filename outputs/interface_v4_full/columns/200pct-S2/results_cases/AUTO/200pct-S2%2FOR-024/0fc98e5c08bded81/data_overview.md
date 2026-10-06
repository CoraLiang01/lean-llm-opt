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
- **FixedCost and Capacity:** Each facility ID (S1, S2, S3) is matched with its FixedCost.
- **Demand:** Each customer ID (C1, C2, C3) is matched with its Demand.
- **Cost-matrix:** Each (Facility, Customer) pair is explicitly listed as above.

**Note:** No explicit capacity is provided for facilities; this is unresolved evidence, not zero.

---

This preserves all required data and structure for modeling the two-dimensional shipment decision as described.