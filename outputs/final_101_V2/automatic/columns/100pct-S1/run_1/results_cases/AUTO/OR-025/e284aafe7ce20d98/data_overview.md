Below is the complete retrieval of all relevant data from the provided context, preserving all facility IDs, customer IDs, fixed costs, demands, and the cost-matrix axes, with explicit source-row positions and original orientation. No data is omitted, transposed, or inferred beyond what is present.

---

### 1. Fixed Costs (from "fixed_cost.csv")

| Facility ID | archive_revision_number | FixedCost | document_page_count | Source-row position |
|-------------|------------------------|-----------|--------------------|--------------------|
| S1          | 2                      | 105.97    | 8                  | 3                  |
| S2          | 1                      | 85.31     | 12                 | 4                  |

---

### 2. Transportation Costs (from "transportation_costs.csv")

| Facility ID | record_display_theme | document_page_count | archive_revision_number | C1 (to Customer C1) | C2 (to Customer C2) | Source-row position |
|-------------|---------------------|--------------------|------------------------|---------------------|---------------------|--------------------|
| S1          | Olive               | 2                  | 1                      | 2358.39             | 1492.08             | 5                  |
| S2          | Olive               | 6                  | 1                      | 0.07                | 52.32               | 6                  |

---

### 3. Demand (from "demand.csv")

| Customer ID | archive_revision_number | Demand | document_page_count | Source-row position |
|-------------|------------------------|--------|--------------------|--------------------|
| C1          | 3                      | 144    | 4                  | 1                  |
| C2          | 5                      | 216    | 6                  | 2                  |

---

#### Matrix Orientation (as per source):

- **Facilities (Suppliers):** S1, S2
- **Customers (Supermarkets):** C1, C2
- **Transportation cost matrix shape:** 2 (facilities) × 2 (customers), with explicit mapping:
    - S1 → C1: 2358.39
    - S1 → C2: 1492.08
    - S2 → C1: 0.07
    - S2 → C2: 52.32

#### Fixed Costs:
- S1: 105.97
- S2: 85.31

#### Demands:
- C1: 144
- C2: 216

---

**All identifiers, values, and axes are preserved as in the original data.**