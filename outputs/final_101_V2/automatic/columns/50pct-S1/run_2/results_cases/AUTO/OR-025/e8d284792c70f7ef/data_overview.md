Below is the complete retrieval of all data from the specified files, preserving all identifiers, values, and source-row positions. No data is omitted, transposed, or inferred beyond what is present.

---

### fixed_cost.csv

| Facility ID | archive_revision_number | FixedCost |
|-------------|------------------------|-----------|
| S1          | 2                      | 105.97    |
| S2          | 1                      | 85.31     |

---

### transportation_costs.csv

| Facility ID | record_display_theme | archive_revision_number | Customer ID | Cost      | Source Row Position |
|-------------|---------------------|------------------------|-------------|-----------|--------------------|
| S1          | Olive               | 1                      | C1          | 2358.39   | S1, C1             |
| S1          | Olive               | 1                      | C2          | 1492.08   | S1, C2             |
| S2          | Olive               | 1                      | C1          | 0.07      | S2, C1             |
| S2          | Olive               | 1                      | C2          | 52.32     | S2, C2             |

---

### demand.csv

| Customer ID | archive_revision_number | Demand | Source Row Position |
|-------------|------------------------|--------|--------------------|
| C1          | 3                      | 144    | C1                 |
| C2          | 5                      | 216    | C2                 |

---

**All facility IDs, customer IDs, fixed costs, and demand values are preserved as in the source. The cost-matrix axis is explicitly matched between facilities (S1, S2) and customers (C1, C2), with no transposition or inference. No capacity data is present; its absence is unresolved evidence, not zero.**