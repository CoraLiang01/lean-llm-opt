Below is a complete retrieval of all relevant data from the provided context, preserving all facility (supplier) and customer (store) IDs, fixed costs, demands, and the full cost matrix, with explicit source-row positions and no inferred or omitted data. No transposition, truncation, or zero-filling has been performed. All data is presented as it appears in the source.

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

#### 3.1. MOUNT AYR (Hybrid, Batch 304, Row 6)
| To Customer (Store) | Cost   |
|---------------------|--------|
| CLARINDA            | 694.68 |
| FORT MADISON        | 17.48  |
| SIOUX CITY          | 20.07  |
| TOLEDO              | 199.02 |
| BANCROFT            | 1685.53|

#### 3.2. WAUKEE (Paper, Batch 302, Row 1)
| To Customer (Store) | Cost   |
|---------------------|--------|
| CLARINDA            | 15.13  |
| FORT MADISON        | 1.50   |
| SIOUX CITY          | 1.43   |
| TOLEDO              | 27.88  |
| BANCROFT            | 90.69  |

#### 3.3. WAVERLY (Digital, Batch 301, Row 6)
| To Customer (Store) | Cost   |
|---------------------|--------|
| CLARINDA            | 2.34   |
| FORT MADISON        | 349.34 |
| SIOUX CITY          | 246.60 |
| TOLEDO              | 41.30  |
| BANCROFT            | 78.73  |

#### 3.4. PELLA (Paper, Batch 301, Row 1)
| To Customer (Store) | Cost   |
|---------------------|--------|
| CLARINDA            | 1181.60|
| FORT MADISON        | 1458.53|
| SIOUX CITY          | 1646.36|
| TOLEDO              | 1924.55|
| BANCROFT            | 38.93  |

#### 3.5. DES MOINES (Digital, Batch 304, Row 4)
| To Customer (Store) | Cost   |
|---------------------|--------|
| CLARINDA            | 1030.80|
| FORT MADISON        | 43.48  |
| SIOUX CITY          | 932.43 |
| TOLEDO              | 55.39  |
| BANCROFT            | 103.84 |

---

### 4. Document Page Counts (for reference, from all sources)

| Facility/Customer | Document Page Count | Source Row |
|-------------------|--------------------|------------|
| MOUNT AYR         | 4                  | 4          |
| WAUKEE            | 2                  | 6          |
| WAVERLY           | 4                  | 1          |
| PELLA             | 6                  | 4          |
| DES MOINES        | 12                 | 3          |
| Customer_1        | 4                  | 1          |
| Customer_2        | 6                  | 2          |
| Customer_3        | 6                  | 2          |
| Customer_4        | 12                 | 5          |
| Customer_5        | 8                  | 5          |

---

**Note:**  
- All facility and customer IDs, fixed costs, demands, and transportation costs are preserved as in the source, with explicit row positions.
- No capacity data is present in the provided context; thus, capacity is unresolved.
- The cost matrix is presented in its original orientation, with each supplier’s costs to each customer/store.
- No data has been omitted, inferred, or altered.

If you need the data in a specific format (e.g., matrix or table for optimization modeling), please specify.