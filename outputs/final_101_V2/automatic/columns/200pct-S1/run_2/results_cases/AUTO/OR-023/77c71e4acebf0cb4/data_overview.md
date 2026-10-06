Below is the complete retrieval of all relevant data from the provided context, preserving all facility (supplier) and customer (store) identifiers, fixed costs, demands, and the full cost matrix for transportation. Each value is shown with its source-row position and original orientation. No data is omitted, transposed, or inferred beyond what is present.

---

### 1. Fixed Costs (from "fixed_cost.csv")

| Facility (Supplier) | FixedCost | Source: archive_batch_number | archive_revision_number | document_page_count | record_view_count |
|---------------------|-----------|-----------------------------|------------------------|--------------------|------------------|
| MOUNT AYR           | 96.58     | 304                         | 4                      | 4                  | 58               |
| WAUKEE              | 94.06     | 302                         | 6                      | 2                  | 12               |
| WAVERLY             | 94.37     | 301                         | 1                      | 4                  | 76               |
| PELLA               | 82.88     | 301                         | 4                      | 6                  | 43               |
| DES MOINES          | 94.96     | 302                         | 3                      | 12                 | 76               |

---

### 2. Demand (from "demand.csv")

| Customer (Store) | Demand | Source: archive_batch_number | archive_revision_number | document_page_count | record_view_count |
|------------------|--------|-----------------------------|------------------------|--------------------|------------------|
| Customer_1       | 2397   | 302                         | 1                      | 4                  | 58               |
| Customer_2       | 1889   | 301                         | 2                      | 6                  | 43               |
| Customer_3       | 2518   | 301                         | 2                      | 6                  | 76               |
| Customer_4       | 3218   | 301                         | 5                      | 12                 | 12               |
| Customer_5       | 1813   | 301                         | 5                      | 8                  | 43               |

---

### 3. Transportation Costs (from "transportation_costs.csv")

#### Each row is a supplier (facility), each column is a store (customer). Values are per-unit transportation costs.

#### (A) MOUNT AYR (archive_batch_number: 304, archive_revision_number: 6, document_page_count: 2, record_view_count: 58)
| Store         | Cost   |
|---------------|--------|
| CLARINDA      | 694.68 |
| FORT MADISON  | 17.48  |
| SIOUX CITY    | 20.07  |
| TOLEDO        | 199.02 |
| BANCROFT      | 1685.53|

#### (B) WAUKEE (archive_batch_number: 302, archive_revision_number: 1, document_page_count: 16, record_view_count: 58)
| Store         | Cost   |
|---------------|--------|
| CLARINDA      | 15.13  |
| FORT MADISON  | 1.50   |
| SIOUX CITY    | 1.43   |
| TOLEDO        | 27.88  |
| BANCROFT      | 90.69  |

#### (C) WAVERLY (archive_batch_number: 301, archive_revision_number: 6, document_page_count: 8, record_view_count: 12)
| Store         | Cost   |
|---------------|--------|
| CLARINDA      | 2.34   |
| FORT MADISON  | 349.34 |
| SIOUX CITY    | 246.60 |
| TOLEDO        | 41.30  |
| BANCROFT      | 78.73  |

#### (D) PELLA (archive_batch_number: 301, archive_revision_number: 1, document_page_count: 16, record_view_count: 76)
| Store         | Cost    |
|---------------|---------|
| CLARINDA      | 1181.60 |
| FORT MADISON  | 1458.53 |
| SIOUX CITY    | 1646.36 |
| TOLEDO        | 1924.55 |
| BANCROFT      | 38.93   |

#### (E) DES MOINES (archive_batch_number: 304, archive_revision_number: 4, document_page_count: 2, record_view_count: 27)
| Store         | Cost    |
|---------------|---------|
| CLARINDA      | 1030.80 |
| FORT MADISON  | 43.48   |
| SIOUX CITY    | 932.43  |
| TOLEDO        | 55.39   |
| BANCROFT      | 103.84  |

---

### 4. Additional Notes

- No explicit supplier capacity is provided in the data; capacity is unresolved evidence (not zero).
- All identifiers and values are preserved as in the source.
- No data is omitted, transposed, or inferred beyond what is present.

---

**Summary Table:**

#### Facilities (Suppliers):  
- MOUNT AYR  
- WAUKEE  
- WAVERLY  
- PELLA  
- DES MOINES  

#### Customers (Stores):  
- CLARINDA  
- FORT MADISON  
- SIOUX CITY  
- TOLEDO  
- BANCROFT  

#### Fixed Costs:  
- As listed above, per facility.

#### Demands:  
- As listed above, per customer.

#### Transportation Cost Matrix:  
- As listed above, each facility-to-customer pair.

---

**All data from fixed_cost.csv, transportation_costs.csv, and demand.csv is retrieved and preserved in original orientation and detail.**