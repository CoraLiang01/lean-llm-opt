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

| Facility ID (Unnamed: 0) | document_page_count | archive_revision_number | fixed_costs |
|--------------------------|--------------------|------------------------|-------------|
| S1                       | 8                  | 2                      | 102.33      |
| S2                       | 6                  | 4                      | 94.92       |
| S3                       | 8                  | 2                      | 91.83       |

---

### 3. transportation_costs.csv

| Facility ID (Unnamed: 0) | archive_storage_medium | C1      | C2      | C3     | record_display_theme | archive_revision_number | document_page_count |
|--------------------------|-----------------------|---------|---------|--------|---------------------|------------------------|--------------------|
| S1                       | Paper                 | 1506.22 | 70.9    | 8.44   | Olive               | 4                      | 6                  |
| S2                       | Paper                 | 1732.65 | 1780.72 | 567.44 | Azure               | 4                      | 8                  |
| S3                       | Digital               | 115.66  | 100.76  | 64.68  | Amber               | 1                      | 4                  |

---

**Preserved Identifiers and Values:**

- **Facility IDs:** S1, S2, S3
- **Customer IDs:** C1, C2, C3
- **Fixed Costs:** S1: 102.33, S2: 94.92, S3: 91.83
- **Demands:** C1: 1083, C2: 776, C3: 16214
- **Transportation Cost Matrix (Facility → Customer):**
    - S1 → C1: 1506.22, S1 → C2: 70.9, S1 → C3: 8.44
    - S2 → C1: 1732.65, S2 → C2: 1780.72, S2 → C3: 567.44
    - S3 → C1: 115.66, S3 → C2: 100.76, S3 → C3: 64.68

**Source Row Positions and Context:**
- All data is preserved as per original row and column orientation.
- No capacity data is present; capacity is unresolved.
- No extra axes or inferred products are introduced.

---

**End of Retrieval**