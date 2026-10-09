Below is the complete retrieval of all relevant data from the provided context, preserving all facility (supplier) and customer (store) identifiers, fixed costs, demands, and the transportation cost matrix. All source-row positions, axis orientations, and original shapes are retained. No transposition, truncation, padding, or inference is performed. Each table is labeled with its source and structure.

---

### 1. Fixed Costs (from "fixed_cost.csv")

| Facility (Supplier) | FixedCost | Source Row Position |
|---------------------|-----------|--------------------|
| MOUNT AYR           | 96.58     | archive_batch_number: 304, archive_revision_number: 4 |
| WAUKEE              | 94.06     | archive_batch_number: 302, archive_revision_number: 6 |
| WAVERLY             | 94.37     | archive_batch_number: 301, archive_revision_number: 1 |
| PELLA               | 82.88     | archive_batch_number: 301, archive_revision_number: 4 |
| DES MOINES          | 94.96     | archive_batch_number: 302, archive_revision_number: 3 |

---

### 2. Demand (from "demand.csv")

| Customer (Store) | Demand | Source Row Position |
|------------------|--------|--------------------|
| Customer_1       | 2397   | archive_batch_number: 302, archive_revision_number: 1 |
| Customer_2       | 1889   | archive_batch_number: 301, archive_revision_number: 2 |
| Customer_3       | 2518   | archive_batch_number: 301, archive_revision_number: 2 |
| Customer_4       | 3218   | archive_batch_number: 301, archive_revision_number: 5 |
| Customer_5       | 1813   | archive_batch_number: 301, archive_revision_number: 5 |

---

### 3. Transportation Costs (from "transportation_costs.csv")

#### (Each row: Facility/Supplier; Each column: Customer/Store; Values: Cost per unit)

**A. MOUNT AYR (archive_batch_number: 304, archive_revision_number: 6)**
| Customer/Store | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|----------------|----------|--------------|------------|--------|----------|
| Cost           | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |

**B. WAUKEE (archive_batch_number: 302, archive_revision_number: 1)**
| Customer/Store | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|----------------|----------|--------------|------------|--------|----------|
| Cost           | 15.13    | 1.5          | 1.43       | 27.88  | 90.69    |

**C. WAVERLY (archive_batch_number: 301, archive_revision_number: 6)**
| Customer/Store | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|----------------|----------|--------------|------------|--------|----------|
| Cost           | 2.34     | 349.34       | 246.6      | 41.3   | 78.73    |

**D. PELLA (archive_batch_number: 301, archive_revision_number: 1)**
| Customer/Store | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|----------------|----------|--------------|------------|--------|----------|
| Cost           | 1181.6   | 1458.53      | 1646.36    | 1924.55| 38.93    |

**E. DES MOINES (archive_batch_number: 304, archive_revision_number: 4)**
| Customer/Store | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|----------------|----------|--------------|------------|--------|----------|
| Cost           | 1030.8   | 43.48        | 932.43     | 55.39  | 103.84   |

---

#### Notes:
- All facility and customer IDs, costs, and demands are preserved as in the source.
- No capacity data is present in the provided context; thus, capacity is unresolved for all facilities.
- The transportation cost matrix is presented in its original orientation: each row is a facility, each column is a customer.
- All source-row positions and archive identifiers are included for traceability.

---

**End of data retrieval.**