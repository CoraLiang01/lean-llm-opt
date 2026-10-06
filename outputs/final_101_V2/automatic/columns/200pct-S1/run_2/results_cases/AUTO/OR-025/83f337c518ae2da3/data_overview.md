Below is the complete retrieval of all relevant data from the provided context, preserving all facility IDs, customer IDs, fixed costs, demand, and the cost-matrix axes, with explicit source-row positions and original orientation. No data is omitted, transposed, or inferred beyond what is present.

---

### 1. fixed_cost.csv

| Facility ID | archive_batch_number | archive_revision_number | FixedCost | record_view_count | document_page_count | Source-row position |
|-------------|---------------------|------------------------|-----------|-------------------|--------------------|--------------------|
| S1          | 305                 | 2                      | 105.97    | 58                | 8                  | 3                  |
| S2          | 303                 | 1                      | 85.31     | 27                | 12                 | 4                  |

---

### 2. demand.csv

| Customer ID | archive_batch_number | archive_revision_number | Demand | record_view_count | document_page_count | Source-row position |
|-------------|---------------------|------------------------|--------|-------------------|--------------------|--------------------|
| C1          | 305                 | 3                      | 144    | 58                | 4                  | 1                  |
| C2          | 303                 | 5                      | 216    | 27                | 6                  | 2                  |

---

### 3. transportation_costs.csv

#### Cost matrix (Facilities as rows, Customers as columns):

| Facility (row) | Customer (column) | Cost   | archive_revision_number | record_view_count | document_page_count | Source-row position |
|----------------|-------------------|--------|------------------------|-------------------|--------------------|--------------------|
| S1             | C1                | 2358.39| 1                      | 58                | 2                  | 5                  |
| S1             | C2                | 1492.08| 1                      | 58                | 2                  | 5                  |
| S2             | C1                | 0.07   | 1                      | 76                | 6                  | 6                  |
| S2             | C2                | 52.32  | 1                      | 76                | 6                  | 6                  |

---

**All identifiers, values, and axes are preserved as in the original data. No additional products, commodities, or axes are inferred.**