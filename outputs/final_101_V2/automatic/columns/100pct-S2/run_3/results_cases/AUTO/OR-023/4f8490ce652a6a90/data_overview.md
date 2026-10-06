Below is the complete retrieval of all relevant data from the provided context, preserving all facility (supplier) and customer (store) identifiers, fixed costs, demand, and the transportation cost matrix. All source-row positions and axis orientations are retained as in the original data.

---

### 1. Fixed Cost Data (from "fixed_cost.csv")

| Facility (Supplier) | Facility ID (Unnamed: 2) | Reception Desk Count (2025 Q4) | Staff Count | Fixed Cost |
|---------------------|-------------------------|-------------------------------|-------------|------------|
| Facility_1          | MOUNT AYR               | 3                             | 12          | 96.58      |
| Facility_2          | WAUKEE                  | 3                             | 12          | 94.06      |
| Facility_3          | WAVERLY                 | 2                             | 20          | 94.37      |
| Facility_4          | PELLA                   | 2                             | 8           | 82.88      |
| Facility_5          | DES MOINES              | 3                             | 50          | 94.96      |

---

### 2. Demand Data (from "demand.csv")

| Customer (Store) | Customer ID | Demand |
|------------------|-------------|--------|
| Customer_1       | Customer_1  | 2397   |
| Customer_2       | Customer_2  | 1889   |
| Customer_3       | Customer_3  | 2518   |
| Customer_4       | Customer_4  | 3218   |
| Customer_5       | Customer_5  | 1813   |

---

### 3. Transportation Cost Matrix (from "transportation_costs.csv")

#### Matrix: Facilities (rows) × Customers (columns)
- Row: Facility (Supplier) [Facility ID = Unnamed: 2]
- Column: Customer (Store) [Customer ID = CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT]
- Each value: Transportation cost per unit from supplier to store

| Facility (Supplier) | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|---------------------|----------|--------------|------------|--------|----------|
| MOUNT AYR           | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE              | 15.13    | 1.50         | 1.43       | 27.88  | 90.69    |
| WAVERLY             | 2.34     | 349.34       | 246.60     | 41.30  | 78.73    |
| PELLA               | 1181.60  | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES          | 1030.80  | 43.48        | 932.43     | 55.39  | 103.84   |

- Source-row positions and axis orientation are as in the original data.

---

### 4. Summary of Preserved Identifiers and Data

- **Facility IDs:** MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES
- **Customer IDs:** CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT
- **Fixed Costs:** As above, matched to each facility
- **Demands:** As above, matched to each customer
- **Transportation Costs:** As above, matrix shape and orientation preserved

---

**No data has been transposed, truncated, padded, zero-filled, or inferred beyond the original context. All identifiers and values are preserved as required for a two-dimensional shipment decision model.**