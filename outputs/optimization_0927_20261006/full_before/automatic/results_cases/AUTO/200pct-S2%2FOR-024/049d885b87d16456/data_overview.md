Here is the complete retrieval of all relevant data from the described files, preserving all identifiers, values, and source-row positions. No narrowing or inference beyond the original query is performed.

---

### 1. Demand Data (`demand.csv`)

| Customer ID | Demand | Source Row |
|-------------|--------|------------|
| C1          | 1083   | 1          |
| C2          | 776    | 2          |
| C3          | 16214  | 3          |

---

### 2. Fixed Cost Data (`fixed_cost.csv`)

| Facility ID | Fixed Cost | Source Row |
|-------------|------------|------------|
| S1          | 102.33     | 1          |
| S2          | 94.92      | 2          |
| S3          | 91.83      | 3          |

---

### 3. Transportation Cost Matrix (`transportation_costs.csv`)

| Facility ID | Customer ID | Transportation Cost per Unit | Source Row (Facility) | Source Column (Customer) |
|-------------|-------------|-----------------------------|-----------------------|--------------------------|
| S1          | C1          | 1506.22                     | 1                     | 1                        |
| S1          | C2          | 70.9                        | 1                     | 2                        |
| S1          | C3          | 8.44                        | 1                     | 3                        |
| S2          | C1          | 1732.65                     | 2                     | 1                        |
| S2          | C2          | 1780.72                     | 2                     | 2                        |
| S2          | C3          | 567.44                      | 2                     | 3                        |
| S3          | C1          | 115.66                      | 3                     | 1                        |
| S3          | C2          | 100.76                      | 3                     | 2                        |
| S3          | C3          | 64.68                       | 3                     | 3                        |

---

**All identifiers (facility IDs, customer IDs), fixed costs, demands, and the full cost matrix are preserved with their original orientation and source positions. No data has been omitted, transposed, or inferred.**