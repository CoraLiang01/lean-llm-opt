Retrieved Data:

**From fixed_cost.csv:**
| Facility ID (Unnamed: 0) | FixedCost (current period) | FixedCost (previous period) | FixedCost (two periods ago) | Source Row Position |
|--------------------------|---------------------------|-----------------------------|-----------------------------|--------------------|
| S1                       | 105.97                    | 95.90285                    | 117.6267                    | 1                  |
| S2                       | 85.31                     | 95.692227                   | 70.798769                   | 2                  |

**From transportation_costs.csv:**
| Facility ID (Unnamed: 0) | Customer ID | Cost (current period) | Cost (previous period) | Source Row Position |
|--------------------------|-------------|----------------------|-----------------------|--------------------|
| S1                       | C1          | 2358.39              | 2727.478035           | 1                  |
| S1                       | C2          | 1492.08              | 1479.397320           | 1                  |
| S2                       | C1          | 0.07                 | 0.071652              | 2                  |
| S2                       | C2          | 52.32                | 57.39504              | 2                  |

**From demand.csv:**
| Customer ID | Demand (current period) | Demand (previous period) | Demand (two periods ago) | Source Row Position |
|-------------|------------------------|--------------------------|--------------------------|--------------------|
| C1          | 144                    | 172                      | 143                      | 1                  |
| C2          | 216                    | 230                      | 195                      | 2                  |

**Preserved Structure:**
- Facility IDs: S1, S2
- Customer IDs: C1, C2
- FixedCost and transportation cost matrices retain original orientation (rows: facilities, columns: customers).
- Demand vector indexed by customer.
- No capacity data present; capacity is unresolved.
- All source row positions and original identifiers are preserved.