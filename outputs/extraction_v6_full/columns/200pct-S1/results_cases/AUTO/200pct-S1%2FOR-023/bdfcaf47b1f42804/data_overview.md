Below is the complete retrieval of all relevant data from the provided context, preserving all facility (supplier) and customer (store) identifiers, fixed costs, demands, and the full cost matrix for transportation. All source-row positions, axis orientations, and original shapes are retained. No transposition, truncation, padding, or inference of extra axes is performed.

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

#### Axis Orientation:
- **Rows:** Facilities (Suppliers)  
- **Columns:** Customers (Stores)

#### Cost Matrix (per unit):

| Facility \ Customer | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT | Source Row Position |
|---------------------|----------|--------------|------------|--------|----------|--------------------|
| MOUNT AYR           | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  | archive_batch_number: 304, archive_revision_number: 6 |
| WAUKEE              | 15.13    | 1.50         | 1.43       | 27.88  | 90.69    | archive_batch_number: 302, archive_revision_number: 1 |
| WAVERLY             | 2.34     | 349.34       | 246.60     | 41.30  | 78.73    | archive_batch_number: 301, archive_revision_number: 6 |
| PELLA               | 1181.60  | 1458.53      | 1646.36    | 1924.55| 38.93    | archive_batch_number: 301, archive_revision_number: 1 |
| DES MOINES          | 1030.80  | 43.48        | 932.43     | 55.39  | 103.84   | archive_batch_number: 304, archive_revision_number: 4 |

---

#### Notes:
- Each facility (supplier) is uniquely identified by its city name.
- Each customer (store) is uniquely identified by its customer ID or city name as per the cost matrix.
- Fixed costs are associated with each facility.
- Demand is specified for each customer.
- The transportation cost matrix is provided for all facility-customer pairs, with explicit source-row positions.

---

**All data is preserved as requested, with no narrowing, inference, or omission.**