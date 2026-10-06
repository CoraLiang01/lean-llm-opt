Below is the complete retrieval of all data from the provided context, preserving all identifiers, values, and source-row positions. The data is organized by file and retains the original axis orientation and shape.

---

### fixed_cost.csv

| Facility ID (Unnamed: 0) | FixedCost |
|--------------------------|-----------|
| S1                       | 98.88     |
| S2                       | 99.73     |
| S3                       | 94.01     |
| S4                       | 93.77     |
| S5                       | 107.59    |
| S6                       | 112.65    |

---

### demand.csv

| Customer ID | Demand |
|-------------|--------|
| C1          | 216    |
| C2          | 216    |
| C3          | 216    |
| C4          | 144    |
| C5          | 144    |
| C6          | 144    |

---

### transportation_costs.csv

| Facility ID (Unnamed: 0) | C1     | C2     | C3      | C4      | C5     | C6     |
|--------------------------|--------|--------|---------|---------|--------|--------|
| S1                       | 0.08   | 52.33  | 73.57   | 1237.33 | 0.07   | 112.16 |
| S2                       | 46.02  | 175.23 | 2026.83 | 299.89  | 966.53 | 1590.42|
| S3                       | 1031.74| 78.13  | 99.02   | 277.07  | 884.45 | 1800.86|
| S4                       | 868.75 | 94.2   | 1776.34 | 285.48  | 868.85 | 86.55  |
| S5                       | 1577   | 760.15 | 2090.19 | 43.2    | 1577.12| 1095.17|
| S6                       | 49.14  | 4.33   | 2079.57 | 277.04  | 1032.01| 1543.49|

---

**Notes:**
- Facility IDs: S1, S2, S3, S4, S5, S6
- Customer IDs: C1, C2, C3, C4, C5, C6
- All fixed costs, demands, and transportation costs are preserved as provided.
- No capacity data is present; capacity is unresolved.
- The cost matrix is facility (row) × customer (column), as in the source.

This data supports a two-dimensional shipment decision model for the described facility location and transportation problem.