Below is the complete retrieval of all relevant data from the provided context, preserving all facility (supplier) and customer (store) identifiers, fixed costs, demands, and the transportation cost matrix. All source-row positions, axis orientations, and original shapes are retained. No transposition, truncation, padding, or inference of extra axes is performed. All data from the three files—fixed_cost.csv, transportation_costs.csv, and demand.csv—are included.

---

### 1. Fixed Costs (fixed_cost.csv)

| Facility (Supplier) | archive_batch_number | archive_revision_number | FixedCost | Source-row position |
|---------------------|---------------------|------------------------|-----------|--------------------|
| MOUNT AYR           | 304                 | 4                      | 96.58     | 1                  |
| WAUKEE              | 302                 | 6                      | 94.06     | 2                  |
| WAVERLY             | 301                 | 1                      | 94.37     | 3                  |
| PELLA               | 301                 | 4                      | 82.88     | 4                  |
| DES MOINES          | 302                 | 3                      | 94.96     | 5                  |

---

### 2. Demand (demand.csv)

| Customer (Store) | archive_batch_number | archive_revision_number | Demand | document_page_count | Source-row position |
|------------------|---------------------|------------------------|--------|--------------------|--------------------|
| Customer_1       | 302                 | 1                      | 2397   | 4                  | 1                  |
| Customer_2       | 301                 | 2                      | 1889   | 6                  | 2                  |
| Customer_3       | 301                 | 2                      | 2518   | 6                  | 3                  |
| Customer_4       | 301                 | 5                      | 3218   | 12                 | 4                  |
| Customer_5       | 301                 | 5                      | 1813   | 8                  | 5                  |

---

### 3. Transportation Costs (transportation_costs.csv)

#### Axis orientation:
- **Rows:** Facilities (Suppliers) — [MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES]
- **Columns:** Customers (Stores) — [CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT]

##### MOUNT AYR (archive_batch_number: 304, archive_revision_number: 6, document_page_count: 2)
| Customer      | Transportation Cost | Source-row position |
|---------------|--------------------|--------------------|
| CLARINDA      | 694.68             | 1                  |
| FORT MADISON  | 17.48              | 1                  |
| SIOUX CITY    | 20.07              | 1                  |
| TOLEDO        | 199.02             | 1                  |
| BANCROFT      | 1685.53            | 1                  |

##### WAUKEE (archive_batch_number: 302, archive_revision_number: 1, document_page_count: 16)
| Customer      | Transportation Cost | Source-row position |
|---------------|--------------------|--------------------|
| CLARINDA      | 15.13              | 2                  |
| FORT MADISON  | 1.50               | 2                  |
| SIOUX CITY    | 1.43               | 2                  |
| TOLEDO        | 27.88              | 2                  |
| BANCROFT      | 90.69               | 2                  |

##### WAVERLY (archive_batch_number: 301, archive_revision_number: 6, document_page_count: 8)
| Customer      | Transportation Cost | Source-row position |
|---------------|--------------------|--------------------|
| CLARINDA      | 2.34               | 3                  |
| FORT MADISON  | 349.34             | 3                  |
| SIOUX CITY    | 246.60             | 3                  |
| TOLEDO        | 41.30              | 3                  |
| BANCROFT      | 78.73              | 3                  |

##### PELLA (archive_batch_number: 301, archive_revision_number: 1, document_page_count: 16)
| Customer      | Transportation Cost | Source-row position |
|---------------|--------------------|--------------------|
| CLARINDA      | 1181.60            | 4                  |
| FORT MADISON  | 1458.53            | 4                  |
| SIOUX CITY    | 1646.36            | 4                  |
| TOLEDO        | 1924.55            | 4                  |
| BANCROFT      | 38.93              | 4                  |

##### DES MOINES (archive_batch_number: 304, archive_revision_number: 4, document_page_count: 2)
| Customer      | Transportation Cost | Source-row position |
|---------------|--------------------|--------------------|
| CLARINDA      | 1030.80            | 5                  |
| FORT MADISON  | 43.48              | 5                  |
| SIOUX CITY    | 932.43             | 5                  |
| TOLEDO        | 55.39              | 5                  |
| BANCROFT      | 103.84             | 5                  |

---

#### Matrix Form (Facilities × Customers):

|                | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|----------------|----------|--------------|------------|--------|----------|
| MOUNT AYR      | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE         | 15.13    | 1.50         | 1.43       | 27.88  | 90.69    |
| WAVERLY        | 2.34     | 349.34       | 246.60     | 41.30  | 78.73    |
| PELLA          | 1181.60  | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES     | 1030.80  | 43.48        | 932.43     | 55.39  | 103.84   |

---

**All identifiers, values, and source-row positions are preserved as required. No data has been omitted, transposed, or inferred.**