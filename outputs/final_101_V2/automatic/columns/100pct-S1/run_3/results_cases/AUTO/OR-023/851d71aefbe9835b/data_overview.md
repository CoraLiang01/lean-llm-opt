Below is a complete retrieval of all relevant data from the provided context, preserving all facility IDs, customer IDs, fixed costs, demands, and the full cost matrix, with explicit source-row positions and no inferred or omitted data. Each table is labeled with its source and retains its original orientation and shape.

---

### 1. Fixed Costs (from "fixed_cost.csv")

| Facility (Unnamed: 2) | archive_revision_number | document_page_count | FixedCost |
|-----------------------|------------------------|--------------------|-----------|
| MOUNT AYR             | 4                      | 4                  | 96.58     |
| WAUKEE                | 6                      | 2                  | 94.06     |
| WAVERLY               | 1                      | 4                  | 94.37     |
| PELLA                 | 4                      | 6                  | 82.88     |
| DES MOINES            | 3                      | 12                 | 94.96     |

---

### 2. Demand (from "demand.csv")

| Customer    | archive_revision_number | document_page_count | Demand |
|-------------|------------------------|--------------------|--------|
| Customer_1  | 1                      | 4                  | 2397   |
| Customer_2  | 2                      | 6                  | 1889   |
| Customer_3  | 2                      | 6                  | 2518   |
| Customer_4  | 5                      | 12                 | 3218   |
| Customer_5  | 5                      | 8                  | 1813   |

---

### 3. Transportation Costs (from "transportation_costs.csv")

#### 3.1. MOUNT AYR (Hybrid, archive_batch_number: 304, archive_revision_number: 6, document_page_count: 2)
| Facility   | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|------------|----------|--------------|------------|--------|----------|
| MOUNT AYR  | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |

#### 3.2. WAUKEE (Paper, archive_batch_number: 302, archive_revision_number: 1, document_page_count: 16)
| Facility | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|----------|----------|--------------|------------|--------|----------|
| WAUKEE   | 15.13    | 1.5          | 1.43       | 27.88  | 90.69    |

#### 3.3. WAVERLY (Digital, archive_batch_number: 301, archive_revision_number: 6, document_page_count: 8)
| Facility | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|----------|----------|--------------|------------|--------|----------|
| WAVERLY  | 2.34     | 349.34       | 246.6      | 41.3   | 78.73    |

#### 3.4. PELLA (Paper, archive_batch_number: 301, archive_revision_number: 1, document_page_count: 16)
| Facility | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|----------|----------|--------------|------------|--------|----------|
| PELLA    | 1181.6   | 1458.53      | 1646.36    | 1924.55| 38.93    |

#### 3.5. DES MOINES (Digital, archive_batch_number: 304, archive_revision_number: 4, document_page_count: 2)
| Facility    | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|-------------|----------|--------------|------------|--------|----------|
| DES MOINES  | 1030.8   | 43.48        | 932.43     | 55.39  | 103.84   |

---

**Notes:**
- All facility and customer identifiers are preserved as in the source.
- Fixed costs are associated with each facility as listed.
- Demand is listed per customer.
- The transportation cost matrix is provided for each facility to each customer location (CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT), with each row corresponding to a facility and each column to a customer location.
- No capacities are stated in the data; absence of capacity is unresolved, not zero.
- No data is omitted, transposed, or inferred beyond the original context.

---

**End of data retrieval.**