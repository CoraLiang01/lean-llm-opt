Below is the complete retrieval of all relevant data from the provided context, preserving all facility (supplier) and customer (store) identifiers, fixed costs, demand, and the transportation cost matrix. Each entry is shown with its source-row position and all relevant axes, as required.

---

### 1. Fixed Costs (from "fixed_cost.csv")
| Facility (Supplier) | Source Row | Fixed Cost |
|---------------------|------------|------------|
| MOUNT AYR           | 304, rev 4 | 96.58      |
| WAUKEE              | 302, rev 6 | 94.06      |
| WAVERLY             | 301, rev 1 | 94.37      |
| PELLA               | 301, rev 4 | 82.88      |
| DES MOINES          | 302, rev 3 | 94.96      |

---

### 2. Demand (from "demand.csv")
| Customer (Store) | Source Row | Demand |
|------------------|------------|--------|
| Customer_1       | 302, rev 1 | 2397   |
| Customer_2       | 301, rev 2 | 1889   |
| Customer_3       | 301, rev 2 | 2518   |
| Customer_4       | 301, rev 5 | 3218   |
| Customer_5       | 301, rev 5 | 1813   |

---

### 3. Transportation Costs (from "transportation_costs.csv")
#### Each table below is a row from the cost matrix, with the facility (supplier) as the row and the customers (stores) as the columns.

#### a. Facility: MOUNT AYR (archive_batch_number: 304, rev 6)
| Customer/Store | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|----------------|----------|--------------|------------|--------|----------|
| Cost           | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |

#### b. Facility: WAUKEE (archive_batch_number: 302, rev 1)
| Customer/Store | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|----------------|----------|--------------|------------|--------|----------|
| Cost           | 15.13    | 1.50         | 1.43       | 27.88  | 90.69    |

#### c. Facility: WAVERLY (archive_batch_number: 301, rev 6)
| Customer/Store | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|----------------|----------|--------------|------------|--------|----------|
| Cost           | 2.34     | 349.34       | 246.60     | 41.30  | 78.73    |

#### d. Facility: PELLA (archive_batch_number: 301, rev 1)
| Customer/Store | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|----------------|----------|--------------|------------|--------|----------|
| Cost           | 1181.60  | 1458.53      | 1646.36    | 1924.55| 38.93    |

#### e. Facility: DES MOINES (archive_batch_number: 304, rev 4)
| Customer/Store | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|----------------|----------|--------------|------------|--------|----------|
| Cost           | 1030.80  | 43.48        | 932.43     | 55.39  | 103.84   |

---

### 4. Additional Notes
- No explicit capacity values are present for any facility; thus, capacity is unresolved evidence (not zero).
- All identifiers (facility and customer names) and matrix axes are preserved as in the source.
- No data has been transposed, truncated, padded, or inferred beyond what is present.

---

**Summary Table of All Data:**

#### Facilities (Suppliers) and Fixed Costs
| Facility      | Fixed Cost |
|---------------|------------|
| MOUNT AYR     | 96.58      |
| WAUKEE        | 94.06      |
| WAVERLY       | 94.37      |
| PELLA         | 82.88      |
| DES MOINES    | 94.96      |

#### Customers (Stores) and Demand
| Customer      | Demand     |
|---------------|------------|
| Customer_1    | 2397       |
| Customer_2    | 1889       |
| Customer_3    | 2518       |
| Customer_4    | 3218       |
| Customer_5    | 1813       |

#### Transportation Cost Matrix (Facility → Customer)
| Facility \ Customer | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|---------------------|----------|--------------|------------|--------|----------|
| MOUNT AYR           | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE              | 15.13    | 1.50         | 1.43       | 27.88  | 90.69    |
| WAVERLY             | 2.34     | 349.34       | 246.60     | 41.30  | 78.73    |
| PELLA               | 1181.60  | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES          | 1030.80  | 43.48        | 932.43     | 55.39  | 103.84   |

---

**All data is preserved in original orientation and detail, as required.**