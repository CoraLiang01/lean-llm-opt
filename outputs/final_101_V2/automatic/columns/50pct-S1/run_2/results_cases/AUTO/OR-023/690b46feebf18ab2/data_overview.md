Below is the complete retrieval of all data from the provided context, preserving all facility IDs, customer IDs, fixed costs, demand, and the full transportation cost matrix. All identifiers and values are retained as in the source, with explicit mapping of facilities (suppliers) and customers (stores).

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

#### Matrix: Cost from each Facility (row) to each Customer (column)
- **Row ID:** Facility (Supplier)
- **Column IDs:** CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT
- **Source orientation and shape retained.**

| Facility (Supplier) | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT | Source Row (archive_revision_number) | document_page_count | record_display_theme |
|---------------------|----------|--------------|------------|--------|----------|--------------------------------------|--------------------|---------------------|
| MOUNT AYR           | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  | 6                                    | 2                  | Slate               |
| WAUKEE              | 15.13    | 1.5          | 1.43       | 27.88  | 90.69    | 1                                    | 16                 | Azure               |
| WAVERLY             | 2.34     | 349.34       | 246.6      | 41.3   | 78.73    | 6                                    | 8                  | Slate               |
| PELLA               | 1181.6   | 1458.53      | 1646.36    | 1924.55| 38.93    | 1                                    | 16                 | Amber               |
| DES MOINES          | 1030.8   | 43.48        | 932.43     | 55.39  | 103.84   | 4                                    | 2                  | Azure               |

---

#### Notes:
- **Facility IDs:** MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES
- **Customer IDs:** CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT
- **Fixed costs** are associated with each facility as listed.
- **Demand** is provided for Customer_1 through Customer_5 (mapping to store IDs; mapping to CLARINDA, etc., is not explicitly provided in the context).
- **Transportation cost matrix** is provided for each facility to each customer (CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT), with all values and source metadata preserved.

---

**All data is retrieved as requested, with no narrowing or omission.**