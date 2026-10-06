Below is the complete retrieval of all relevant data from the provided context, preserving all facility (supplier) and customer (store) identifiers, fixed costs, and the transportation cost matrix, as well as demand values. Each value is shown with its source-row position and original orientation. No data is omitted, transposed, or inferred beyond what is present.

---

### 1. Fixed Costs (from `fixed_cost.csv`)
| Row | Facility (Supplier) | FixedCost (current period) | FixedCost (previous period) | FixedCost (two periods ago) |
|-----|---------------------|---------------------------|-----------------------------|-----------------------------|
| 1   | MOUNT AYR           | 96.58                     | 101.457290                  | 89.809742                   |
| 2   | WAUKEE              | 94.06                     | 112.034866                  | 108.319496                  |
| 3   | WAVERLY             | 94.37                     | 86.06544                    | 99.201744                   |
| 4   | PELLA               | 82.88                     | 89.526976                   | 81.620224                   |
| 5   | DES MOINES          | 94.96                     | 110.894288                  | 85.938800                   |

---

### 2. Demand (from `demand.csv`)
| Row | Customer (Store) | Demand (current period) | Demand (previous period) | Demand (two periods ago) |
|-----|------------------|------------------------|--------------------------|--------------------------|
| 1   | Customer_1       | 2397                   | 2025                     | 2634                     |
| 2   | Customer_2       | 1889                   | 1729                     | 1966                     |
| 3   | Customer_3       | 2518                   | 2280                     | 2219                     |
| 4   | Customer_4       | 3218                   | 3008                     | 3823                     |
| 5   | Customer_5       | 1813                   | 1814                     | 1759                     |

---

### 3. Transportation Costs (from `transportation_costs.csv`)
Each row is for a supplier (facility), each column is a store (customer). All values are per unit cost.

#### 3.1. MOUNT AYR (Row 1)
| To (Customer) | SIOUX CITY | CLARINDA | FORT MADISON | TOLEDO | BANCROFT |
|---------------|------------|----------|--------------|--------|----------|
| Current       | 20.07      | 694.68   | 17.48        | 199.02 | 1685.53  |
| Previous      | 16.70      | 832.71   | 19.01        | 197.77 | -        |
| Two periods ago service status: Seasonal |

#### 3.2. WAUKEE (Row 2)
| To (Customer) | SIOUX CITY | CLARINDA | FORT MADISON | TOLEDO | BANCROFT |
|---------------|------------|----------|--------------|--------|----------|
| Current       | 1.43       | 15.13    | 1.5          | 27.88  | 90.69    |
| Previous      | 1.48       | 13.74    | 1.76         | 33.12  | -        |
| Two periods ago service status: Trial |

#### 3.3. WAVERLY (Row 3)
| To (Customer) | SIOUX CITY | CLARINDA | FORT MADISON | TOLEDO | BANCROFT |
|---------------|------------|----------|--------------|--------|----------|
| Current       | 246.6      | 2.34     | 349.34       | 41.3   | 78.73    |
| Previous      | 221.99     | 1.98     | 342.04       | 44.99  | -        |
| Two periods ago service status: Suspended |

#### 3.4. PELLA (Row 4)
| To (Customer) | SIOUX CITY | CLARINDA | FORT MADISON | TOLEDO | BANCROFT |
|---------------|------------|----------|--------------|--------|----------|
| Current       | 1646.36    | 1181.6   | 1458.53      | 1924.55| 38.93    |
| Previous      | 1676.16    | 960.05   | 1520.66      | 1693.41| -        |
| Two periods ago service status: Seasonal |

#### 3.5. DES MOINES (Row 5)
| To (Customer) | SIOUX CITY | CLARINDA | FORT MADISON | TOLEDO | BANCROFT |
|---------------|------------|----------|--------------|--------|----------|
| Current       | 932.43     | 1030.8   | 43.48        | 55.39  | 103.84   |
| Previous      | 944.83     | 1093.99  | 43.73        | 53.21  | -        |
| Two periods ago service status: Trial |

---

**Notes:**
- All facility and customer IDs are preserved as in the source.
- Fixed costs are per facility, per period.
- Demand is per customer, per period.
- Transportation costs are per facility-to-customer pair, per period (current period shown; previous periods available as above).
- No capacity data is present in the provided context.
- No data is omitted, inferred, or transposed.

---

**This is the complete data set as requested, suitable for modeling the two-dimensional shipment decision (x_{ij}) and supplier activation (y_i) for the Iowa Department of Commerce liquor supply problem.**