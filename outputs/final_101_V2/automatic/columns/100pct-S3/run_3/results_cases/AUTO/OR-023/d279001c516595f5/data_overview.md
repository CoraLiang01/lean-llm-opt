Below is the complete retrieval of all data from the provided context, preserving all facility (supplier) and customer (store) identifiers, fixed costs, and the transportation cost matrix, as well as all demand values. Each value is shown with its source-row position and original orientation. No data is omitted, transposed, or inferred beyond what is present.

---

### fixed_cost.csv

| Row | Facility (Supplier) | FixedCost (current period) | FixedCost (previous period) | FixedCost (two periods ago) |
|-----|---------------------|---------------------------|-----------------------------|-----------------------------|
| 1   | MOUNT AYR           | 96.58                     | 101.457290                  | 89.809742                   |
| 2   | WAUKEE              | 94.06                     | 112.034866                  | 108.319496                  |
| 3   | WAVERLY             | 94.37                     | 86.06544                    | 99.201744                   |
| 4   | PELLA               | 82.88                     | 89.526976                   | 81.620224                   |
| 5   | DES MOINES          | 94.96                     | 110.894288                  | 85.938800                   |

---

### demand.csv

| Row | Customer (Store) | Demand (current period) | Demand (previous period) | Demand (two periods ago) |
|-----|------------------|------------------------|--------------------------|--------------------------|
| 1   | Customer_1       | 2397                   | 2025                     | 2634                     |
| 2   | Customer_2       | 1889                   | 1729                     | 1966                     |
| 3   | Customer_3       | 2518                   | 2280                     | 2219                     |
| 4   | Customer_4       | 3218                   | 3008                     | 3823                     |
| 5   | Customer_5       | 1813                   | 1814                     | 1759                     |

---

### transportation_costs.csv

#### Row 1: MOUNT AYR (Supplier)
| To (Customer/Store) | SIOUX CITY | CLARINDA | FORT MADISON | TOLEDO | BANCROFT |
|---------------------|------------|----------|--------------|--------|----------|
| Cost                | 20.07      | 694.68   | 17.48        | 199.02 | 1685.53  |
| Previous period     | 16.702254  | 832.712916 | 19.009500  | 197.766174 | -      |
| Two periods ago     | Seasonal   | -        | -            | -      | -        |

#### Row 2: WAUKEE (Supplier)
| To (Customer/Store) | SIOUX CITY | CLARINDA | FORT MADISON | TOLEDO | BANCROFT |
|---------------------|------------|----------|--------------|--------|----------|
| Cost                | 1.43       | 15.13    | 1.5          | 27.88  | 90.69    |
| Previous period     | 1.480193   | 13.736527| 1.75905      | 33.124228 | -      |
| Two periods ago     | Trial      | -        | -            | -      | -        |

#### Row 3: WAVERLY (Supplier)
| To (Customer/Store) | SIOUX CITY | CLARINDA | FORT MADISON | TOLEDO | BANCROFT |
|---------------------|------------|----------|--------------|--------|----------|
| Cost                | 246.6      | 2.34     | 349.34       | 41.3   | 78.73    |
| Previous period     | 221.98932  | 1.975662 | 342.038794   | 44.98809 | -      |
| Two periods ago     | Suspended  | -        | -            | -      | -        |

#### Row 4: PELLA (Supplier)
| To (Customer/Store) | SIOUX CITY | CLARINDA | FORT MADISON | TOLEDO | BANCROFT |
|---------------------|------------|----------|--------------|--------|----------|
| Cost                | 1646.36    | 1181.6   | 1458.53      | 1924.55| 38.93    |
| Previous period     | 1676.159116| 960.05000| 1520.663378  | 1693.411545 | -   |
| Two periods ago     | Seasonal   | -        | -            | -      | -        |

#### Row 5: DES MOINES (Supplier)
| To (Customer/Store) | SIOUX CITY | CLARINDA | FORT MADISON | TOLEDO | BANCROFT |
|---------------------|------------|----------|--------------|--------|----------|
| Cost                | 932.43     | 1030.8   | 43.48        | 55.39  | 103.84   |
| Previous period     | 944.831319 | 1093.98804| 43.727836   | 53.213173 | -     |
| Two periods ago     | Trial      | -        | -            | -      | -        |

---

#### Notes:
- All facility (supplier) and customer (store) IDs are preserved as in the source.
- Fixed costs are shown for each facility for the current, previous, and two periods ago.
- Demand is shown for each customer for the current, previous, and two periods ago.
- The transportation cost matrix is shown for each supplier to each store, with previous period and two periods ago values where available.
- No capacity data is present in the provided context.
- No data is omitted, transposed, or inferred beyond what is present.

---

**This is the complete data retrieval as requested, suitable for modeling a two-dimensional shipment decision problem as described.**