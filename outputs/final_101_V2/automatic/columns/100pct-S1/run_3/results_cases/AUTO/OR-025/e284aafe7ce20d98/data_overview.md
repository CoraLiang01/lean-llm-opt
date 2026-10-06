Below is the complete retrieval of all relevant data from the provided context, preserving every facility (supplier) ID, customer (supermarket) ID, fixed cost, demand, and the cost-matrix axes, with explicit source-row positions and original orientation. No data is omitted, transposed, or inferred beyond what is present.

---

### 1. Fixed Costs (from "fixed_cost.csv")

| Facility ID (Unnamed: 0) | archive_revision_number | FixedCost | document_page_count | Source-row position |
|-------------------------|------------------------|-----------|--------------------|--------------------|
| S1                      | 2                      | 105.97    | 8                  | 3                  |
| S2                      | 1                      | 85.31     | 12                 | 4                  |

---

### 2. Transportation Costs (from "transportation_costs.csv")

| Facility ID (Unnamed: 0) | record_display_theme | document_page_count | archive_revision_number | C1 (to customer C1) | C2 (to customer C2) | Source-row position |
|-------------------------|---------------------|--------------------|------------------------|---------------------|---------------------|--------------------|
| S1                      | Olive               | 2                  | 1                      | 2358.39             | 1492.08             | 5                  |
| S2                      | Olive               | 6                  | 1                      | 0.07                | 52.32               | 6                  |

---

### 3. Demand (from "demand.csv")

| Customer ID | archive_revision_number | Demand | document_page_count | Source-row position |
|-------------|------------------------|--------|--------------------|--------------------|
| C1          | 3                      | 144    | 4                  | 1                  |
| C2          | 5                      | 216    | 6                  | 2                  |

---

**Summary of preserved axes and identifiers:**

- Facilities (Suppliers): S1, S2
- Customers (Supermarkets): C1, C2
- FixedCost: S1 (105.97), S2 (85.31)
- Demand: C1 (144), C2 (216)
- Transportation cost matrix (Facility → Customer):
    - S1 → C1: 2358.39
    - S1 → C2: 1492.08
    - S2 → C1: 0.07
    - S2 → C2: 52.32

All data is preserved as per the original query, with explicit mapping of facility and customer IDs, and no inferred or omitted values.