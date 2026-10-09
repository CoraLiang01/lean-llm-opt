Below is the complete retrieval of all data from the provided context, preserving all facility IDs, customer IDs, fixed costs, demand, and the transportation cost matrix. All identifiers and values are retained as in the source, with no transposition, truncation, or inference beyond the original data.

---

### fixed_cost.csv

| Facility (Unnamed: 1) | Fixed Opening Cost Previous Period | Fixed Costs |
|-----------------------|------------------------------------|-------------|
| MOUNT AYR             | 101.457290                         | 96.58       |
| WAUKEE                | 112.034866                         | 94.06       |
| WAVERLY               | 86.06544                           | 94.37       |
| PELLA                 | 89.526976                          | 82.88       |
| DES MOINES            | 110.894288000                      | 94.96       |

---

### demand.csv

| Customer    | Demand Previous Period | Demand |
|-------------|-----------------------|--------|
| Customer_1  | 2025                  | 2397   |
| Customer_2  | 1729                  | 1889   |
| Customer_3  | 2280                  | 2518   |
| Customer_4  | 3008                  | 3218   |
| Customer_5  | 1814                  | 1813   |

---

### transportation_costs.csv

#### Source: Each row is a supplier (facility), each column is a store (customer or location).

| Facility (Unnamed: 0) | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|-----------------------|----------|--------------|------------|--------|----------|
| MOUNT AYR             | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE                | 15.13    | 1.5          | 1.43       | 27.88  | 90.69    |
| WAVERLY               | 2.34     | 349.34       | 246.6      | 41.3   | 78.73    |
| PELLA                 | 1181.6   | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES            | 1030.8   | 43.48        | 932.43     | 55.39  | 103.84   |

#### Additional context per facility (from source):

- MOUNT AYR: previous_period_FORT_MADISON: 19.0095, previous_period_CLARINDA: 832.712916, previous_period_service_status: Seasonal
- WAUKEE: previous_period_FORT_MADISON: 1.75905, previous_period_CLARINDA: 13.736527, previous_period_service_status: Trial
- WAVERLY: previous_period_FORT_MADISON: 342.038794, previous_period_CLARINDA: 1.975662, previous_period_service_status: Regular
- PELLA: previous_period_FORT_MADISON: 1520.663378, previous_period_CLARINDA: 960.05, previous_period_service_status: Regular
- DES MOINES: previous_period_FORT_MADISON: 43.727836, previous_period_CLARINDA: 1093.98804, previous_period_service_status: Regular

---

**All data is preserved as in the original context, with explicit facility and customer IDs, fixed costs, demand, and the full cost matrix.**