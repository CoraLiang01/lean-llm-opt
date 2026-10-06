**Retrieved Data**

---

### demand.csv

| customer | demand_previous_period | demand |
|----------|-----------------------|--------|
| C1       | 1006                  | 1083   |
| C2       | 842                   | 776    |
| C3       | 16770                 | 16214  |

---

### fixed_cost.csv

| Unnamed: 0 (Facility ID) | fixed_opening_cost_previous_period | fixed_costs |
|--------------------------|------------------------------------|-------------|
| S1                       | 86.734908                          | 102.33      |
| S2                       | 78.02424                           | 94.92       |
| S3                       | 105.51267                          | 91.83       |

---

### transportation_costs.csv

| Unnamed: 0 (Facility ID) | C1      | C2      | C3     | previous_period_service_status | previous_period_C1 |
|--------------------------|---------|---------|--------|-------------------------------|--------------------|
| S1                       | 1506.22 | 70.9    | 8.44   | Suspended                     | 1364.785942        |
| S2                       | 1732.65 | 1780.72 | 567.44 | Seasonal                      | 1958.414295        |
| S3                       | 115.66  | 100.76  | 64.68  | Regular                       | 131.574816         |

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

**Source Row Positions and Orientation:**
- All data is preserved in its original row/column orientation as provided. No transposition, truncation, or inference has been performed. No capacity data is present; capacity is unresolved. No extra axes have been added.

**End of Retrieval**