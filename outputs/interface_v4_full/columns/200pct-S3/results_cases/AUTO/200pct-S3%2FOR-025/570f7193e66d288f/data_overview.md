Below is the complete retrieval of all data from the three requested sources, with all identifiers, values, and matrix axes preserved as in the original context.

---

### fixed_cost.csv

| Facility ID (Unnamed: 0) | four_periods_ago_fixed_costs | three_periods_ago_fixed_costs | two_periods_ago_fixed_costs | fixed_opening_cost_previous_period | fixed_costs |
|--------------------------|------------------------------|-------------------------------|-----------------------------|------------------------------------|-------------|
| S1                       | 112.720289                   | 103.967167                    | 117.6267                    | 95.90285                           | 105.97      |
| S2                       | 70.022448                    | 81.803759                     | 70.798769                   | 95.692227                          | 85.31       |

---

### transportation_costs.csv

| Facility ID (Unnamed: 1) | three_periods_ago_service_status | two_periods_ago_service_status | previous_period_service_status | previous_period_C1 | previous_period_C2 | two_periods_ago_C1 | two_periods_ago_C2 | C1      | C2      |
|--------------------------|----------------------------------|-------------------------------|-------------------------------|--------------------|--------------------|--------------------|--------------------|---------|---------|
| S1                       | Regular                          | Suspended                     | Regular                       | 2727.478035        | 1479.397320        | 2434.094319        | -                  | 2358.39 | 1492.08 |
| S2                       | Trial                            | Suspended                     | Regular                       | 0.0716520000000    | 57.39504           | 0.0588980000000    | -                  | 0.07000000000000001 | 52.32   |

---

### demand.csv

| Customer | four_periods_ago_demand | three_periods_ago_demand | two_periods_ago_demand | demand_previous_period | demand |
|----------|------------------------|-------------------------|------------------------|-----------------------|--------|
| C1       | 153                    | 147                     | 143                    | 172                   | 144    |
| C2       | 179                    | 235                     | 195                    | 230                   | 216    |

---

**Notes on structure:**
- Facility IDs: S1, S2
- Customer IDs: C1, C2
- Fixed costs are per facility (S1, S2) and provided for several periods; the most recent is under "fixed_costs".
- Transportation costs are per facility-customer pair, with the most recent under columns "C1" and "C2".
- Demand is per customer, with the most recent under "demand".

**Matrix axes:**
- Transportation cost matrix: rows = facilities (S1, S2), columns = customers (C1, C2)
- Demand vector: indexed by customer (C1, C2)
- Fixed cost vector: indexed by facility (S1, S2)

**All data is preserved as in the source, with no transposition, truncation, or inference.**