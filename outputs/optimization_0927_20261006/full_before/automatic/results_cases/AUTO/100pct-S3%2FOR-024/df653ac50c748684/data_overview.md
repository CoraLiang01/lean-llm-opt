Here is the complete retrieval of all data from the three files as requested, with all identifiers and values preserved. No narrowing or inference beyond the original query is performed.

---

### demand.csv

| Customer | Demand | Demand Previous Period | Two Periods Ago Demand | Source Row Position |
|----------|--------|-----------------------|-----------------------|--------------------|
| C1       | 1083   | 1006                  | 900                   | 1                  |
| C2       | 776    | 842                   | 622                   | 2                  |
| C3       | 16214  | 16770                 | 13681                 | 3                  |

---

### fixed_cost.csv

| Facility ID | Fixed Cost | Fixed Opening Cost Previous Period | Two Periods Ago Fixed Costs | Source Row Position |
|-------------|------------|------------------------------------|----------------------------|--------------------|
| S1          | 102.33     | 86.734908                          | 122.120622                 | 1                  |
| S2          | 94.92      | 78.02424                           | 90.107556                  | 2                  |
| S3          | 91.83      | 105.51267                          | 97.24797                   | 3                  |

---

### transportation_costs.csv

| Facility ID | Customer | Cost | Previous Period Cost | Two Periods Ago Service Status | Previous Period Service Status | Source Row Position |
|-------------|----------|------|---------------------|-------------------------------|-------------------------------|--------------------|
| S1          | C1       | 1506.22 | 1364.785942      | Seasonal                      | Suspended                     | 1                  |
| S1          | C2       | 70.9    | 70.99217          | Seasonal                      | Suspended                     | 1                  |
| S1          | C3       | 8.44    | -                 | Seasonal                      | Suspended                     | 1                  |
| S2          | C1       | 1732.65 | 1958.414295       | Trial                         | Seasonal                      | 2                  |
| S2          | C2       | 1780.72 | 1932.793488       | Trial                         | Seasonal                      | 2                  |
| S2          | C3       | 567.44  | -                 | Trial                         | Seasonal                      | 2                  |
| S3          | C1       | 115.66  | 131.574816        | Trial                         | Regular                       | 3                  |
| S3          | C2       | 100.76  | 91.258332         | Trial                         | Regular                       | 3                  |
| S3          | C3       | 64.68   | -                 | Trial                         | Regular                       | 3                  |

---

**Notes:**
- All facility IDs (S1, S2, S3) and customer IDs (C1, C2, C3) are preserved.
- Fixed costs are matched to facility IDs.
- Demand is matched to customer IDs.
- The transportation cost matrix is provided with explicit facility-customer pairs, with all available historical context.
- No capacity data is present in the provided context.
- All source row positions are retained for traceability.
- No data is omitted, transposed, or inferred beyond the original query.