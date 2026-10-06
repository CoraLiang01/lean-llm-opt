Here is the complete retrieval of all data from 'demand.csv', 'fixed_cost.csv', and 'transportation_costs.csv', with all identifiers and values preserved, as requested:

---

### demand.csv

| archive_revision_number | customer | demand |
|------------------------|----------|--------|
| 4                      | C1       | 1083   |  ← source-row 1
| 5                      | C2       | 776    |  ← source-row 2
| 2                      | C3       | 16214  |  ← source-row 3

---

### fixed_cost.csv

| archive_revision_number | Unnamed: 0 (Facility ID) | fixed_costs |
|------------------------|--------------------------|-------------|
| 2                      | S1                       | 102.33      |  ← source-row 4
| 4                      | S2                       | 94.92       |  ← source-row 5
| 2                      | S3                       | 91.83       |  ← source-row 6

---

### transportation_costs.csv

| Unnamed: 0 (Facility ID) | C1      | C2      | C3     | record_display_theme | archive_revision_number |
|--------------------------|---------|---------|--------|---------------------|------------------------|
| S1                       | 1506.22 | 70.9    | 8.44   | Olive               | 4                      |  ← source-row 7
| S2                       | 1732.65 | 1780.72 | 567.44 | Azure               | 4                      |  ← source-row 8
| S3                       | 115.66  | 100.76  | 64.68  | Amber               | 1                      |  ← source-row 9

---

#### Axis and ID mapping (preserved):

- Facility IDs: S1, S2, S3 (from fixed_cost.csv and transportation_costs.csv, matching by Unnamed: 0)
- Customer IDs: C1, C2, C3 (from demand.csv and transportation_costs.csv columns)
- FixedCost: S1 (102.33), S2 (94.92), S3 (91.83)
- Demand: C1 (1083), C2 (776), C3 (16214)
- Cost-matrix: Each row is a facility (S1, S2, S3), each column is a customer (C1, C2, C3), values as above.

No capacity data is present; this is unresolved evidence, not zero.

---

**All data is preserved in original orientation and with explicit source-row positions.**