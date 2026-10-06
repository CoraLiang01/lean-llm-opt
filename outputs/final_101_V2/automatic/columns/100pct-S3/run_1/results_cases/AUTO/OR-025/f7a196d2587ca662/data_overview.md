Below is the complete retrieval of all relevant data from the provided context, preserving all facility IDs, customer IDs, fixed costs, demand, and the cost-matrix axes, with explicit matching of facilities and customers. No data is omitted, transposed, or inferred beyond what is present.

---

### fixed_cost.csv

| Facility ID (Unnamed: 0) | FixedCost (current period) | FixedCost (previous period) | FixedCost (two periods ago) |
|--------------------------|---------------------------|-----------------------------|-----------------------------|
| S1                       | 105.97                    | 95.90285                    | 117.6267                    |
| S2                       | 85.31                     | 95.692227                   | 70.798769                   |

---

### transportation_costs.csv

| Facility ID (Unnamed: 0) | Service Status (previous period) | Previous Period: C1 | Previous Period: C2 | Current Period: C1 | Current Period: C2 |
|--------------------------|----------------------------------|---------------------|---------------------|--------------------|--------------------|
| S1                       | Regular                          | 2727.478035         | 1479.397320         | 2358.39            | 1492.08            |
| S2                       | Regular                          | 0.071652            | 57.39504            | 0.07               | 52.32              |

*Note: The cost matrix is oriented as [Facility, Customer].*

---

### demand.csv

| Customer ID | Demand (current period) | Demand (previous period) | Demand (two periods ago) |
|-------------|------------------------|--------------------------|--------------------------|
| C1          | 144                    | 172                      | 143                      |
| C2          | 216                    | 230                      | 195                      |

---

**All identifiers and values are preserved as in the source. No additional inference, padding, or transformation has been performed.**