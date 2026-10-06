Below is the complete retrieval of all relevant data from the provided context, preserving all facility IDs, customer IDs, fixed costs, demands, and the cost-matrix axes, with explicit source-row positions and original orientations. No transposition, truncation, or inference is performed.

---

### 1. Fixed Costs (from "fixed_cost.csv")

| Facility ID | archive_batch_number | archive_revision_number | FixedCost | Source-row position |
|-------------|---------------------|------------------------|-----------|--------------------|
| S1          | 305                 | 2                      | 105.97    | 3                  |
| S2          | 303                 | 1                      | 85.31     | 4                  |

---

### 2. Demand (from "demand.csv")

| Customer ID | archive_batch_number | archive_revision_number | Demand | Source-row position |
|-------------|---------------------|------------------------|--------|--------------------|
| C1          | 305                 | 3                      | 144    | 1                  |
| C2          | 303                 | 5                      | 216    | 2                  |

---

### 3. Transportation Costs (from "transportation_costs.csv")

#### Source-row position 5 (archive_revision_number: 1, archive_batch_number: 305, S1)
- Facility: S1
- Customers: C1, C2
- Cost-matrix (row: S1, columns: C1, C2):

|        | C1      | C2      |
|--------|---------|---------|
| **S1** | 2358.39 | 1492.08 |

#### Source-row position 6 (archive_revision_number: 1, archive_batch_number: 303, S2)
- Facility: S2
- Customers: C1, C2
- Cost-matrix (row: S2, columns: C1, C2):

|        | C1   | C2   |
|--------|------|------|
| **S2** | 0.07 | 52.32 |

---

### 4. Facility/Customer/Matrix Axis Summary

- Facilities: S1, S2
- Customers: C1, C2
- FixedCost: S1 (105.97), S2 (85.31)
- Demand: C1 (144), C2 (216)
- Transportation cost matrix (facility-to-customer, as rows: S1, S2; columns: C1, C2):

|        | C1      | C2      |
|--------|---------|---------|
| **S1** | 2358.39 | 1492.08 |
| **S2** | 0.07    | 52.32   |

---

**All data is preserved as per the original query, with explicit identifiers and source-row positions.**