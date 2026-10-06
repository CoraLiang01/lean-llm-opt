Below is the complete retrieval of all data from the provided context, preserving all facility IDs, customer IDs, fixed costs, demand, and the full cost matrix, with explicit source-row positions and identifiers. No data is omitted, transposed, or inferred beyond the original context.

---

### 1. Fixed Costs (from `fixed_cost.csv`)

| Facility (Supplier) | Fixed Cost | Source Row (archive_revision_number) |
|---------------------|------------|--------------------------------------|
| MOUNT AYR           | 96.58      | 4                                    |
| WAUKEE              | 94.06      | 6                                    |
| WAVERLY             | 94.37      | 1                                    |
| PELLA               | 82.88      | 4                                    |
| DES MOINES          | 94.96      | 3                                    |

---

### 2. Demand (from `demand.csv`)

| Customer (Store) | Demand | Source Row (archive_revision_number) |
|------------------|--------|--------------------------------------|
| Customer_1       | 2397   | 1                                    |
| Customer_2       | 1889   | 2                                    |
| Customer_3       | 2518   | 2                                    |
| Customer_4       | 3218   | 5                                    |
| Customer_5       | 1813   | 5                                    |

---

### 3. Transportation Costs (from `transportation_costs.csv`)

#### Each row is a supplier (facility), each column is a customer (store). The values are the per-unit transportation costs.

##### a. MOUNT AYR (archive_revision_number: 6, document_page_count: 2, record_display_theme: Slate)
| To Customer      | Cost   |
|------------------|--------|
| CLARINDA         | 694.68 |
| FORT MADISON     | 17.48  |
| SIOUX CITY       | 20.07  |
| TOLEDO           | 199.02 |
| BANCROFT         | 1685.53|

##### b. WAUKEE (archive_revision_number: 1, document_page_count: 16, record_display_theme: Azure)
| To Customer      | Cost   |
|------------------|--------|
| CLARINDA         | 15.13  |
| FORT MADISON     | 1.50   |
| SIOUX CITY       | 1.43   |
| TOLEDO           | 27.88  |
| BANCROFT         | 90.69  |

##### c. WAVERLY (archive_revision_number: 6, document_page_count: 8, record_display_theme: Slate)
| To Customer      | Cost   |
|------------------|--------|
| CLARINDA         | 2.34   |
| FORT MADISON     | 349.34 |
| SIOUX CITY       | 246.60 |
| TOLEDO           | 41.30  |
| BANCROFT         | 78.73  |

##### d. PELLA (archive_revision_number: 1, document_page_count: 16, record_display_theme: Amber)
| To Customer      | Cost   |
|------------------|--------|
| CLARINDA         | 1181.60|
| FORT MADISON     | 1458.53|
| SIOUX CITY       | 1646.36|
| TOLEDO           | 1924.55|
| BANCROFT         | 38.93  |

##### e. DES MOINES (archive_revision_number: 4, document_page_count: 2, record_display_theme: Azure)
| To Customer      | Cost   |
|------------------|--------|
| CLARINDA         | 1030.80|
| FORT MADISON     | 43.48  |
| SIOUX CITY       | 932.43 |
| TOLEDO           | 55.39  |
| BANCROFT         | 103.84 |

---

### Notes on Data Structure

- **Facility IDs**: MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES
- **Customer IDs**: CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT
- **Fixed Costs**: Explicitly matched to each facility.
- **Demand**: Provided for Customer_1 through Customer_5 (mapping to CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT is implied by order but not explicitly stated in the context).
- **Transportation Cost Matrix**: Each facility row lists costs to each customer column, with all original identifiers and values preserved.

---

**All data from the three files is retrieved and preserved as requested, with all identifiers, values, and source-row positions intact.**