Below is the complete retrieval of all data from the provided context, preserving all identifiers and values from the equivalent of fixed_cost.csv, transportation_costs.csv, and demand.csv. Each facility (supplier) and customer (store) is kept with its associated fixed cost, demand, and transportation cost matrix, with all source-row positions and identifiers intact.

---

### 1. Fixed Cost Data (fixed_cost.csv)

| Facility (Supplier) | Fixed Cost (Current Period) | Four Periods Ago | Three Periods Ago | Two Periods Ago | Previous Period | Source Row |
|---------------------|----------------------------|------------------|-------------------|-----------------|----------------|------------|
| MOUNT AYR           | 96.58                      | 95.633516        | 109.637616        | 89.809742       | 101.457290     | 1          |
| WAUKEE              | 94.06                      | 111.846746       | 97.211010         | 108.319496      | 112.034866     | 2          |
| WAVERLY             | 94.37                      | 87.509301        | 112.904268        | 99.201744       | 86.06544       | 3          |
| PELLA               | 82.88                      | 78.288448        | 92.527232         | 81.620224       | 89.526976      | 4          |
| DES MOINES          | 94.96                      | 77.487360        | 109.109040        | 85.938800       | 110.894288     | 5          |

---

### 2. Demand Data (demand.csv)

| Customer (Store) | Demand (Current Period) | Four Periods Ago | Three Periods Ago | Two Periods Ago | Previous Period | Source Row |
|------------------|------------------------|------------------|-------------------|-----------------|----------------|------------|
| Customer_1       | 2397                   | 2316             | 2781              | 2634            | 2025           | 1          |
| Customer_2       | 1889                   | 2106             | 1886              | 1966            | 1729           | 2          |
| Customer_3       | 2518                   | 2128             | 2138              | 2219            | 2280           | 3          |
| Customer_4       | 3218                   | 3380             | 3161              | 3823            | 3008           | 4          |
| Customer_5       | 1813                   | 1931             | 2059              | 1759            | 1814           | 5          |

---

### 3. Transportation Cost Matrix (transportation_costs.csv)

#### (Each table below is for a supplier, with costs to each customer/store. Source-row and orientation preserved.)

#### a. MOUNT AYR (Source Row: 1)
| To Store         | CLARINDA   | FORT MADISON | SIOUX CITY | TOLEDO   | BANCROFT  | Service Status (Current) | Source Row |
|------------------|------------|--------------|------------|----------|-----------|-------------------------|------------|
| Cost             | 694.68     | 17.48        | 20.07      | 199.02   | 1685.53   | Seasonal                | 1          |
| Previous Period  | 832.71     | 19.01        | 16.70      | 197.77   | 1741.99   | Seasonal                |            |
| Two Periods Ago  | 643.90     | 15.93        | 21.17      | -        | -         | Seasonal                |            |
| Three Periods Ago| -          | -            | -          | -        | -         | Regular                 |            |
| Four Periods Ago | -          | -            | -          | -        | -         | Suspended               |            |

#### b. WAUKEE (Source Row: 2)
| To Store         | CLARINDA   | FORT MADISON | SIOUX CITY | TOLEDO   | BANCROFT  | Service Status (Current) | Source Row |
|------------------|------------|--------------|------------|----------|-----------|-------------------------|------------|
| Cost             | 15.13      | 1.5          | 1.43       | 27.88    | 90.69     | Trial                   | 2          |
| Previous Period  | 13.74      | 1.76         | 1.48       | 33.12    | 92.49     | Trial                   |            |
| Two Periods Ago  | 14.81      | 1.70         | 1.55       | -        | -         | Trial                   |            |
| Three Periods Ago| -          | -            | -          | -        | -         | Trial                   |            |
| Four Periods Ago | -          | -            | -          | -        | -         | Suspended               |            |

#### c. WAVERLY (Source Row: 3)
| To Store         | CLARINDA   | FORT MADISON | SIOUX CITY | TOLEDO   | BANCROFT  | Service Status (Current) | Source Row |
|------------------|------------|--------------|------------|----------|-----------|-------------------------|------------|
| Cost             | 2.34       | 349.34       | 246.60     | 41.30    | 78.73     | Suspended               | 3          |
| Previous Period  | 1.98       | 342.04       | 221.99     | 44.99    | 87.13     | Regular                 |            |
| Two Periods Ago  | 2.10       | 351.23       | 223.35     | -        | -         | Suspended               |            |
| Three Periods Ago| -          | -            | -          | -        | -         | Seasonal                |            |
| Four Periods Ago | -          | -            | -          | -        | -         | Regular                 |            |

#### d. PELLA (Source Row: 4)
| To Store         | CLARINDA   | FORT MADISON | SIOUX CITY | TOLEDO   | BANCROFT  | Service Status (Current) | Source Row |
|------------------|------------|--------------|------------|----------|-----------|-------------------------|------------|
| Cost             | 1181.60    | 1458.53      | 1646.36    | 1924.55  | 38.93     | Seasonal                | 4          |
| Previous Period  | 960.05     | 1520.66      | 1676.16    | 1693.41  | 41.21     | Regular                 |            |
| Two Periods Ago  | 1351.99    | 1560.48      | 1695.59    | -        | -         | Seasonal                |            |
| Three Periods Ago| -          | -            | -          | -        | -         | Trial                   |            |
| Four Periods Ago | -          | -            | -          | -        | -         | Seasonal                |            |

#### e. DES MOINES (Source Row: 5)
| To Store         | CLARINDA   | FORT MADISON | SIOUX CITY | TOLEDO   | BANCROFT  | Service Status (Current) | Source Row |
|------------------|------------|--------------|------------|----------|-----------|-------------------------|------------|
| Cost             | 1030.80    | 43.48        | 932.43     | 55.39    | 103.84    | Trial                   | 5          |
| Previous Period  | 1093.99    | 43.73        | 944.83     | 53.21    | 109.09    | Regular                 |            |
| Two Periods Ago  | 877.11     | 48.70        | 862.31     | -        | -         | Trial                   |            |
| Three Periods Ago| -          | -            | -          | -        | -         | Seasonal                |            |
| Four Periods Ago | -          | -            | -          | -        | -         | Trial                   |            |

---

**Note:**  
- All facility and customer IDs, fixed costs, and demand values are preserved as in the source.  
- The transportation cost matrix is presented for each supplier to each store, with all available periods and service statuses.  
- No capacity data is present in the context; thus, capacity is unresolved (not zero).  
- No product axis is present or inferred; all data is for a single product as per the original query.  
- No data is transposed, truncated, or padded.  
- All source-row positions and original orientation are retained.

If you need the data in a specific file format (e.g., CSV, JSON), or a specific matrix layout, please specify.