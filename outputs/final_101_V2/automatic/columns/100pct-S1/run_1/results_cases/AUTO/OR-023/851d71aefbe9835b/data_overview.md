Below is a complete retrieval of all relevant data from the provided context, preserving all facility (supplier) and customer (store) IDs, fixed costs, demands, and the full cost matrix, with explicit source-row positions and no inferred or omitted data. No transposition, truncation, or zero-filling has been performed. Each table is labeled with its source and retains its original orientation.

---

### 1. Fixed Costs (from "fixed_cost.csv")

| Facility (Supplier) | Source Row | Fixed Cost |
|---------------------|------------|------------|
| MOUNT AYR           | 4          | 96.58      |
| WAUKEE              | 6          | 94.06      |
| WAVERLY             | 1          | 94.37      |
| PELLA               | 4          | 82.88      |
| DES MOINES          | 3          | 94.96      |

---

### 2. Demand (from "demand.csv")

| Customer (Store) | Source Row | Demand | Document Page Count |
|------------------|------------|--------|--------------------|
| Customer_1       | 1          | 2397   | 4                  |
| Customer_2       | 2          | 1889   | 6                  |
| Customer_3       | 2          | 2518   | 6                  |
| Customer_4       | 5          | 3218   | 12                 |
| Customer_5       | 5          | 1813   | 8                  |

---

### 3. Transportation Costs (from "transportation_costs.csv")

#### a. MOUNT AYR (Hybrid, Batch 304, Row 6)
| To (Store/City) | Cost   |
|-----------------|--------|
| CLARINDA        | 694.68 |
| FORT MADISON    | 17.48  |
| SIOUX CITY      | 20.07  |
| TOLEDO          | 199.02 |
| BANCROFT        | 1685.53|

#### b. WAUKEE (Paper, Batch 302, Row 1)
| To (Store/City) | Cost   |
|-----------------|--------|
| CLARINDA        | 15.13  |
| FORT MADISON    | 1.50   |
| SIOUX CITY      | 1.43   |
| TOLEDO          | 27.88  |
| BANCROFT        | 90.69  |

#### c. WAVERLY (Digital, Batch 301, Row 6)
| To (Store/City) | Cost   |
|-----------------|--------|
| CLARINDA        | 2.34   |
| FORT MADISON    | 349.34 |
| SIOUX CITY      | 246.60 |
| TOLEDO          | 41.30  |
| BANCROFT        | 78.73  |

#### d. PELLA (Paper, Batch 301, Row 1)
| To (Store/City) | Cost   |
|-----------------|--------|
| CLARINDA        | 1181.60|
| FORT MADISON    | 1458.53|
| SIOUX CITY      | 1646.36|
| TOLEDO          | 1924.55|
| BANCROFT        | 38.93  |

#### e. DES MOINES (Digital, Batch 304, Row 4)
| To (Store/City) | Cost   |
|-----------------|--------|
| CLARINDA        | 1030.80|
| FORT MADISON    | 43.48  |
| SIOUX CITY      | 932.43 |
| TOLEDO          | 55.39  |
| BANCROFT        | 103.84 |

---

### Notes on Data Structure

- **Facilities (Suppliers):** MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES
- **Customers (Stores):** Customer_1, Customer_2, Customer_3, Customer_4, Customer_5
- **Transportation Cost Matrix:** Each supplier has a cost to each city (CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT). The mapping from city to customer is not explicitly provided in the context, so all cost data is preserved as-is.
- **Fixed Costs:** Each supplier has a fixed cost, matched by facility name.
- **Demand:** Each customer has a demand value.

---

**All data is preserved in its original form and orientation, with explicit source-row positions and no inferred or omitted values.**