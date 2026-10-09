Here is the complete retrieval of all relevant data from the provided context, preserving all facility IDs, customer IDs, fixed costs, demand, and the cost-matrix axes, with explicit source-row positions and original orientation:

---

### fixed_cost.csv

| Facility ID | archive_revision_number | FixedCost | document_page_count | Source-row position |
|-------------|------------------------|-----------|--------------------|--------------------|
| S1          | 2                      | 105.97    | 8                  | 3                  |
| S2          | 1                      | 85.31     | 12                 | 4                  |

---

### transportation_costs.csv

| Facility ID | record_display_theme | document_page_count | archive_revision_number | Cost to C1 | Cost to C2 | Source-row position |
|-------------|---------------------|--------------------|------------------------|------------|------------|--------------------|
| S1          | Olive               | 2                  | 1                      | 2358.39    | 1492.08    | 5                  |
| S2          | Olive               | 6                  | 1                      | 0.07       | 52.32      | 6                  |

- The cost-matrix is oriented as: rows = facilities (S1, S2), columns = customers (C1, C2).

---

### demand.csv

| Customer ID | archive_revision_number | Demand | document_page_count | Source-row position |
|-------------|------------------------|--------|--------------------|--------------------|
| C1          | 3                      | 144    | 4                  | 1                  |
| C2          | 5                      | 216    | 6                  | 2                  |

---

**Summary of preserved axes and identifiers:**

- Facilities: S1, S2 (with fixed costs and cost-matrix rows)
- Customers: C1, C2 (with demand and cost-matrix columns)
- FixedCost and cost-matrix row both use the same facility ID.
- Demand and cost-matrix column both use the same customer ID.
- No capacity data is present; capacity is unresolved.
- All data is presented in original orientation and shape, with no transposition, truncation, or inference.

**All data from fixed_cost.csv, transportation_costs.csv, and demand.csv is included as requested.**