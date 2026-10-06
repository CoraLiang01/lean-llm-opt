Below is the complete retrieval of all relevant data from the provided context, preserving all facility IDs, customer IDs, fixed costs, and the cost-matrix axes, as well as demand values. No data is omitted, transposed, or inferred beyond what is present.

---

### fixed_cost.csv

| Facility ID (Unnamed: 0) | fixed_opening_cost_previous_period | fixed_costs |
|--------------------------|------------------------------------|-------------|
| S1                       | 95.90285                           | 105.97      |
| S2                       | 95.692227                          | 85.31       |

---

### transportation_costs.csv

| Facility ID (Unnamed: 0) | previous_period_service_status | previous_period_C1 | C1      | C2     |
|--------------------------|-------------------------------|--------------------|---------|--------|
| S1                       | Regular                       | 2727.478035        | 2358.39 | 1492.08|
| S2                       | Regular                       | 0.0716520000000    | 0.0700  | 52.32  |

- The matrix is oriented as: rows = suppliers (S1, S2), columns = customers (C1, C2).
- Each entry (i, j) gives the per-unit transportation cost from supplier S_i to customer C_j.

---

### demand.csv

| Customer ID | demand_previous_period | demand |
|-------------|-----------------------|--------|
| C1          | 172                   | 144    |
| C2          | 230                   | 216    |

---

**All identifiers and values are preserved as in the source. No additional inference or transformation has been applied.**