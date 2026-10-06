Below is the complete retrieval of all data from the provided context, preserving all identifiers, values, and source-row positions. The data is organized by file as requested: **fixed_cost.csv**, **transportation_costs.csv**, and **demand.csv**. All facility IDs, customer IDs, fixed costs, and demand values are retained, and the cost matrix is presented with explicit facility (supplier) and customer (store) IDs.

---

### fixed_cost.csv

| Row | Facility ID   | FixedCost (current period) | FixedCost (previous period) |
|-----|--------------|---------------------------|-----------------------------|
| 1   | MOUNT AYR    | 96.58                     | 101.457290                  |
| 2   | WAUKEE       | 94.06                     | 112.034866                  |
| 3   | WAVERLY      | 94.37                     | 86.06544                    |
| 4   | PELLA        | 82.88                     | 89.526976                   |
| 5   | DES MOINES   | 94.96                     | 110.894288                  |

---

### transportation_costs.csv

| Row | Facility (Supplier) | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT | previous_period_CLARINDA | previous_period_FORT_MADISON | previous_period_service_status |
|-----|---------------------|----------|--------------|------------|--------|----------|-------------------------|------------------------------|-------------------------------|
| 1   | MOUNT AYR           | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  | 832.712916              | 19.009500                    | Seasonal                      |
| 2   | WAUKEE              | 15.13    | 1.5          | 1.43       | 27.88  | 90.69    | 13.736527               | 1.75905                      | Trial                         |
| 3   | WAVERLY             | 2.34     | 349.34       | 246.6      | 41.3   | 78.73    | 1.975662                | 342.038794                   | Regular                       |
| 4   | PELLA               | 1181.6   | 1458.53      | 1646.36    | 1924.55| 38.93    | 960.05000               | 1520.663378                  | Regular                       |
| 5   | DES MOINES          | 1030.8   | 43.48        | 932.43     | 55.39  | 103.84   | 1093.98804              | 43.727836                    | Regular                       |

---

### demand.csv

| Row | Customer (Store) | Demand (current period) | Demand (previous period) |
|-----|------------------|------------------------|--------------------------|
| 1   | Customer_1       | 2397                   | 2025                     |
| 2   | Customer_2       | 1889                   | 1729                     |
| 3   | Customer_3       | 2518                   | 2280                     |
| 4   | Customer_4       | 3218                   | 3008                     |
| 5   | Customer_5       | 1813                   | 1814                     |

---

**Notes:**
- Facility IDs: MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES
- Customer IDs: Customer_1, Customer_2, Customer_3, Customer_4, Customer_5
- Fixed costs are per facility, as are transportation costs (matrix: facility × customer).
- Demand is per customer (store).
- No explicit capacity data is present in the provided context.

**All data is preserved as in the source, with no transposition, truncation, or inference.**