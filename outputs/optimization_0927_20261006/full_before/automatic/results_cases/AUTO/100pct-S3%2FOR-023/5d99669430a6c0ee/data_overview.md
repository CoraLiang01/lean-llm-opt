Below is the complete retrieval of all relevant data from the provided context, preserving all facility (supplier) and customer (store) identifiers, fixed costs, and the transportation cost matrix, as well as all demand values. Each value is shown with its source-row position and original orientation. No data is omitted, transposed, or inferred beyond what is present.

---

### 1. Fixed Costs (from `fixed_cost.csv`)

| Row | Facility (Supplier) | Fixed Cost (Current Period) | Fixed Cost (Previous Period) | Fixed Cost (Two Periods Ago) |
|-----|---------------------|----------------------------|------------------------------|------------------------------|
| 1   | MOUNT AYR           | 96.58                      | 101.457290                   | 89.809742                    |
| 2   | WAUKEE              | 94.06                      | 112.034866                   | 108.319496                   |
| 3   | WAVERLY             | 94.37                      | 86.06544                     | 99.201744                    |
| 4   | PELLA               | 82.88                      | 89.526976                    | 81.620224                    |
| 5   | DES MOINES          | 94.96                      | 110.894288                   | 85.938800                    |

---

### 2. Transportation Costs (from `transportation_costs.csv`)

#### a. MOUNT AYR (Row 1)
| To (Customer/Store) | Cost (Current Period) | Cost (Previous Period) |
|---------------------|----------------------|-----------------------|
| SIOUX CITY          | 20.07                | 16.702254             |
| CLARINDA            | 694.68               | 832.712916            |
| FORT MADISON        | 17.48                | 19.009500             |
| TOLEDO              | 199.02               | 197.766174            |
| BANCROFT            | 1685.53              | -                     |

#### b. WAUKEE (Row 2)
| To (Customer/Store) | Cost (Current Period) | Cost (Previous Period) |
|---------------------|----------------------|-----------------------|
| SIOUX CITY          | 1.43                 | 1.480193              |
| CLARINDA            | 15.13                | 13.736527             |
| FORT MADISON        | 1.5                  | 1.75905               |
| TOLEDO              | 27.88                | 33.124228             |
| BANCROFT            | 90.69                | -                     |

#### c. WAVERLY (Row 3)
| To (Customer/Store) | Cost (Current Period) | Cost (Previous Period) |
|---------------------|----------------------|-----------------------|
| SIOUX CITY          | 246.6                | 221.98932             |
| CLARINDA            | 2.34                 | 1.975662              |
| FORT MADISON        | 349.34               | 342.038794            |
| TOLEDO              | 41.3                 | 44.98809              |
| BANCROFT            | 78.73                | -                     |

#### d. PELLA (Row 4)
| To (Customer/Store) | Cost (Current Period) | Cost (Previous Period) |
|---------------------|----------------------|-----------------------|
| SIOUX CITY          | 1646.36              | 1676.159116           |
| CLARINDA            | 1181.6               | 960.05000             |
| FORT MADISON        | 1458.53              | 1520.663378           |
| TOLEDO              | 1924.55              | 1693.411545           |
| BANCROFT            | 38.93                | -                     |

#### e. DES MOINES (Row 5)
| To (Customer/Store) | Cost (Current Period) | Cost (Previous Period) |
|---------------------|----------------------|-----------------------|
| SIOUX CITY          | 932.43               | 944.831319            |
| CLARINDA            | 1030.8               | 1093.98804            |
| FORT MADISON        | 43.48                | 43.727836             |
| TOLEDO              | 55.39                | 53.213173             |
| BANCROFT            | 103.84               | -                     |

---

### 3. Demand (from `demand.csv`)

| Row | Customer (Store) | Demand (Current Period) | Demand (Previous Period) | Demand (Two Periods Ago) |
|-----|------------------|------------------------|--------------------------|--------------------------|
| 1   | Customer_1       | 2397                   | 2025                     | 2634                     |
| 2   | Customer_2       | 1889                   | 1729                     | 1966                     |
| 3   | Customer_3       | 2518                   | 2280                     | 2219                     |
| 4   | Customer_4       | 3218                   | 3008                     | 3823                     |
| 5   | Customer_5       | 1813                   | 1814                     | 1759                     |

---

#### Notes:
- All facility (supplier) and customer (store) IDs are preserved as in the source.
- All fixed costs, transportation costs, and demand values are shown for each period as provided.
- No capacity data is present in the context; thus, capacity is unresolved.
- The cost matrix is shown in its original orientation: each supplier (facility) row lists costs to each customer (store).
- No product axis is present or inferred; all data is for a single product as per the query.

---

**This is the complete data retrieval as required for modeling the two-dimensional shipment decision problem.**