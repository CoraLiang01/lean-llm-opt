Below is the complete retrieval of all relevant data from the provided context, preserving all facility IDs, customer IDs, fixed costs, and the cost-matrix axes, as well as demand values. The data is organized by source and retains its original orientation and shape.

---

### fixed_cost.csv

| Facility ID (Unnamed: 0) | FixedCost (current period) | FixedCost (previous period) | FixedCost (two periods ago) |
|--------------------------|---------------------------|-----------------------------|-----------------------------|
| S1                       | 105.97                    | 95.90285                    | 117.6267                    |
| S2                       | 85.31                     | 95.692227                   | 70.798769                   |

---

### transportation_costs.csv

| Facility ID (Unnamed: 0) | previous_period_service_status | previous_period_C1 | previous_period_C2 | C1 (current period) | C2 (current period) |
|--------------------------|-------------------------------|--------------------|--------------------|---------------------|---------------------|
| S1                       | Regular                       | 2727.478035        | 1479.397320        | 2358.39             | 1492.08             |
| S2                       | Regular                       | 0.071652           | 57.39504           | 0.07                | 52.32               |

- The cost matrix is shaped as:  
  - Rows: Facilities (S1, S2)  
  - Columns: Customers (C1, C2)  
  - Values: Per-unit transportation costs from each supplier to each supermarket.

---

### demand.csv

| Customer ID | Demand (current period) | Demand (previous period) | Demand (two periods ago) |
|-------------|------------------------|--------------------------|--------------------------|
| C1          | 144                    | 172                      | 143                      |
| C2          | 216                    | 230                      | 195                      |

---

**All identifiers and values are preserved as in the source data. No transposition, truncation, or inference has been performed.**