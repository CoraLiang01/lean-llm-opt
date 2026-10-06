**Retrieved Data**

---

### demand.csv

| Customer | Demand | Source Row Position |
|----------|--------|--------------------|
| C1       | 1083   | 1                  |
| C2       | 776    | 2                  |
| C3       | 16214  | 3                  |

---

### fixed_cost.csv

| Facility ID | Facility Staff Count | Fixed Cost | Source Row Position |
|-------------|---------------------|------------|--------------------|
| S1          | 20                  | 102.33     | 4                  |
| S2          | 50                  | 94.92      | 5                  |
| S3          | 35                  | 91.83      | 6                  |

---

### transportation_costs.csv

| Facility ID | C1      | C2      | C3     | Operations Region | Annual Inspection Count | Source Row Position |
|-------------|---------|---------|--------|------------------|------------------------|--------------------|
| S1          | 1506.22 | 70.9    | 8.44   | South            | 2                      | 7                  |
| S2          | 1732.65 | 1780.72 | 567.44 | East             | 2                      | 8                  |
| S3          | 115.66  | 100.76  | 64.68  | West             | 4                      | 9                  |

---

**Preserved Identifiers and Matrix Shape**

- Facility IDs: S1, S2, S3
- Customer IDs: C1, C2, C3
- FixedCost and FacilityStaffCount are matched by Facility ID.
- Demand is matched by Customer ID.
- Transportation cost matrix: rows = facilities (S1, S2, S3), columns = customers (C1, C2, C3), as in the source.
- All source row positions are retained for traceability.

**No data has been transposed, truncated, padded, zero-filled, or inferred beyond the original files.**