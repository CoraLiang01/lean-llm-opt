Below is the complete retrieval of all relevant data from the provided context, preserving all facility (supplier) and customer (store) identifiers, fixed costs, demand, and the transportation cost matrix. All source-row positions, axis orientations, and original shapes are retained. No transposition, truncation, padding, or inference of extra axes is performed.

---

### 1. Fixed Costs (from "fixed_cost.csv")

| Facility (Supplier) | Source Row | Fixed Cost |
|---------------------|------------|------------|
| MOUNT AYR           | 304        | 96.58      |
| WAUKEE              | 302        | 94.06      |
| WAVERLY             | 301        | 94.37      |
| PELLA               | 301        | 82.88      |
| DES MOINES          | 302        | 94.96      |

- Each facility is identified by its city name and archive_batch_number (source row).
- Fixed costs are associated with each facility as listed.

---

### 2. Demand (from "demand.csv")

| Customer (Store) | Source Row | Demand |
|------------------|------------|--------|
| Customer_1       | 302        | 2397   |
| Customer_2       | 301        | 1889   |
| Customer_3       | 301        | 2518   |
| Customer_4       | 301        | 3218   |
| Customer_5       | 301        | 1813   |

- Each customer is identified by their customer ID and archive_batch_number (source row).
- Demand is the number of units required by each customer.

---

### 3. Transportation Costs (from "transportation_costs.csv")

#### (a) For archive_batch_number 304 (MOUNT AYR, DES MOINES)

| Facility (Supplier) | Customer (Store) | Transportation Cost | Source Row |
|---------------------|------------------|--------------------|------------|
| MOUNT AYR           | CLARINDA         | 694.68             | 304        |
| MOUNT AYR           | FORT MADISON     | 17.48              | 304        |
| MOUNT AYR           | SIOUX CITY       | 20.07              | 304        |
| MOUNT AYR           | TOLEDO           | 199.02             | 304        |
| MOUNT AYR           | BANCROFT         | 1685.53            | 304        |
| DES MOINES          | CLARINDA         | 1030.8             | 304        |
| DES MOINES          | FORT MADISON     | 43.48              | 304        |
| DES MOINES          | SIOUX CITY       | 932.43             | 304        |
| DES MOINES          | TOLEDO           | 55.39              | 304        |
| DES MOINES          | BANCROFT         | 103.84             | 304        |

#### (b) For archive_batch_number 302 (WAUKEE)

| Facility (Supplier) | Customer (Store) | Transportation Cost | Source Row |
|---------------------|------------------|--------------------|------------|
| WAUKEE              | CLARINDA         | 15.13              | 302        |
| WAUKEE              | FORT MADISON     | 1.50               | 302        |
| WAUKEE              | SIOUX CITY       | 1.43               | 302        |
| WAUKEE              | TOLEDO           | 27.88              | 302        |
| WAUKEE              | BANCROFT         | 90.69              | 302        |

#### (c) For archive_batch_number 301 (WAVERLY, PELLA)

| Facility (Supplier) | Customer (Store) | Transportation Cost | Source Row |
|---------------------|------------------|--------------------|------------|
| WAVERLY             | CLARINDA         | 2.34               | 301        |
| WAVERLY             | FORT MADISON     | 349.34             | 301        |
| WAVERLY             | SIOUX CITY       | 246.60             | 301        |
| WAVERLY             | TOLEDO           | 41.30              | 301        |
| WAVERLY             | BANCROFT         | 78.73              | 301        |
| PELLA               | CLARINDA         | 1181.60            | 301        |
| PELLA               | FORT MADISON     | 1458.53            | 301        |
| PELLA               | SIOUX CITY       | 1646.36            | 301        |
| PELLA               | TOLEDO           | 1924.55            | 301        |
| PELLA               | BANCROFT         | 38.93              | 301        |

---

#### Notes:
- All facility and customer IDs are preserved as in the source.
- Fixed costs and transportation costs are matched to their respective facility and customer IDs, with source-row positions retained.
- No capacity data is present in the provided context; thus, capacity is unresolved.
- The cost matrix is two-dimensional: (Facility, Customer).
- No extra axes or products are inferred; each entry is for a single product as per the original query.

---

**Summary Table of Facilities and Customers:**

| Facility (Supplier) | Fixed Cost | Source Row |
|---------------------|------------|------------|
| MOUNT AYR           | 96.58      | 304        |
| WAUKEE              | 94.06      | 302        |
| WAVERLY             | 94.37      | 301        |
| PELLA               | 82.88      | 301        |
| DES MOINES          | 94.96      | 302/304    |

| Customer (Store) | Demand | Source Row |
|------------------|--------|------------|
| Customer_1       | 2397   | 302        |
| Customer_2       | 1889   | 301        |
| Customer_3       | 2518   | 301        |
| Customer_4       | 3218   | 301        |
| Customer_5       | 1813   | 301        |
| CLARINDA         | —      | —          |
| FORT MADISON     | —      | —          |
| SIOUX CITY       | —      | —          |
| TOLEDO           | —      | —          |
| BANCROFT         | —      | —          |

---

**All data from fixed_cost.csv, transportation_costs.csv, and demand.csv is retrieved and preserved as requested.**