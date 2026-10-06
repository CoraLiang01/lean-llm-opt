Here is the complete retrieval of all data from 'demand.csv', 'fixed_cost.csv', and 'transportation_costs.csv', with all facility IDs, customer IDs, FixedCost, Demand, and cost-matrix axes preserved, including their source-row positions and original orientation:

---

### demand.csv

| customer | demand | document_page_count | archive_revision_number |
|----------|--------|--------------------|------------------------|
| C1       | 1083   | 8                  | 4                      |
| C2       | 776    | 6                  | 5                      |
| C3       | 16214  | 2                  | 2                      |

---

### fixed_cost.csv

| Unnamed: 0 (Facility ID) | fixed_costs | document_page_count | archive_revision_number |
|--------------------------|-------------|--------------------|------------------------|
| S1                       | 102.33      | 8                  | 2                      |
| S2                       | 94.92       | 6                  | 4                      |
| S3                       | 91.83       | 8                  | 2                      |

---

### transportation_costs.csv

| Unnamed: 0 (Facility ID) | archive_storage_medium | C1      | C2      | C3     | record_display_theme | archive_revision_number | document_page_count |
|--------------------------|-----------------------|---------|---------|--------|---------------------|------------------------|--------------------|
| S1                       | Paper                 | 1506.22 | 70.9    | 8.44   | Olive               | 4                      | 6                  |
| S2                       | Paper                 | 1732.65 | 1780.72 | 567.44 | Azure               | 4                      | 8                  |
| S3                       | Digital               | 115.66  | 100.76  | 64.68  | Amber               | 1                      | 4                  |

---

#### Matrix orientation (shipment cost matrix):

- **Rows:** Facilities (S1, S2, S3)
- **Columns:** Customers (C1, C2, C3)
- **Values:** Per-unit transportation cost from facility to customer

#### Facility fixed costs:

- S1: 102.33
- S2: 94.92
- S3: 91.83

#### Customer demands:

- C1: 1083
- C2: 776
- C3: 16214

#### Transportation cost matrix (facility-to-customer):

|        | C1      | C2      | C3     |
|--------|---------|---------|--------|
| S1     | 1506.22 | 70.9    | 8.44   |
| S2     | 1732.65 | 1780.72 | 567.44 |
| S3     | 115.66  | 100.76  | 64.68  |

---

**All identifiers, values, and source-row positions are preserved as requested.**