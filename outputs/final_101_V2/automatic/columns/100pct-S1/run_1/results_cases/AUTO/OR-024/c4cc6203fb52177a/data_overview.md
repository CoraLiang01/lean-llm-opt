**Retrieved Data**

---

### 1. demand.csv

| customer | demand | document_page_count | archive_revision_number |
|----------|--------|--------------------|------------------------|
| C1       | 1083   | 8                  | 4                      |
| C2       | 776    | 6                  | 5                      |
| C3       | 16214  | 2                  | 2                      |

---

### 2. fixed_cost.csv

| facility_id | fixed_costs | document_page_count | archive_revision_number |
|-------------|-------------|--------------------|------------------------|
| S1          | 102.33      | 8                  | 2                      |
| S2          | 94.92       | 6                  | 4                      |
| S3          | 91.83       | 8                  | 2                      |

---

### 3. transportation_costs.csv

| facility_id | archive_storage_medium | C1      | C2      | C3     | record_display_theme | archive_revision_number | document_page_count |
|-------------|-----------------------|---------|---------|--------|---------------------|------------------------|--------------------|
| S1          | Paper                 | 1506.22 | 70.9    | 8.44   | Olive               | 4                      | 6                  |
| S2          | Paper                 | 1732.65 | 1780.72 | 567.44 | Azure               | 4                      | 8                  |
| S3          | Digital               | 115.66  | 100.76  | 64.68  | Amber               | 1                      | 4                  |

---

**Preserved Identifiers and Values:**
- Facility IDs: S1, S2, S3
- Customer IDs: C1, C2, C3
- Fixed Costs: S1 (102.33), S2 (94.92), S3 (91.83)
- Demand: C1 (1083), C2 (776), C3 (16214)
- Transportation Cost Matrix (facility-to-customer, as shown above)
- All source row/column positions and original data orientation retained

**No data has been transposed, truncated, padded, or inferred beyond the original query.**