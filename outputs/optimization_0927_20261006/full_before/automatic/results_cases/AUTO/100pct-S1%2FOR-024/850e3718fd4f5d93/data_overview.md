Here is the complete retrieval of all data from the relevant files, preserving all identifiers, values, and source-row positions. No narrowing or inference beyond the original query is performed.

---

### demand.csv

| customer | demand | document_page_count | archive_revision_number |
|----------|--------|--------------------|------------------------|
| C1       | 1083   | 8                  | 4                      |
| C2       | 776    | 6                  | 5                      |
| C3       | 16214  | 2                  | 2                      |

---

### fixed_cost.csv

| Facility ID (Unnamed: 0) | fixed_costs | document_page_count | archive_revision_number |
|--------------------------|-------------|--------------------|------------------------|
| S1                       | 102.33      | 8                  | 2                      |
| S2                       | 94.92       | 6                  | 4                      |
| S3                       | 91.83       | 8                  | 2                      |

---

### transportation_costs.csv

| Facility ID (Unnamed: 0) | archive_storage_medium | C1      | C2      | C3     | record_display_theme | archive_revision_number | document_page_count |
|--------------------------|-----------------------|---------|---------|--------|---------------------|------------------------|--------------------|
| S1                       | Paper                 | 1506.22 | 70.9    | 8.44   | Olive               | 4                      | 6                  |
| S2                       | Paper                 | 1732.65 | 1780.72 | 567.44 | Azure               | 4                      | 8                  |
| S3                       | Digital               | 115.66  | 100.76  | 64.68  | Amber               | 1                      | 4                  |

---

#### Notes on preservation:
- All facility IDs (S1, S2, S3) and customer IDs (C1, C2, C3) are retained.
- Fixed costs and transportation costs are matched to their respective facility IDs.
- Demand is matched to customer IDs.
- All source-row positions and additional columns (such as document_page_count, archive_revision_number, archive_storage_medium, record_display_theme) are preserved as in the original data.
- No transposition, truncation, padding, zero-filling, or inference of extra axes is performed.

---

**This data supports a two-dimensional shipment decision model with:**
- Facilities: S1, S2, S3 (with fixed costs)
- Customers: C1, C2, C3 (with demands)
- Transportation cost matrix: [S1, S2, S3] × [C1, C2, C3] as shown above

Let me know if you need this in a specific format (e.g., CSV, JSON, or as a mathematical model input)!