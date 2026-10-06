Below is the complete retrieval of all relevant data from the provided context, preserving all facility (supplier) and customer (store) identifiers, fixed costs, demand, and the transportation cost matrix. All source-row positions and axis orientations are retained as in the original data.

---

### 1. Fixed Costs and Facility Data (from "fixed_cost.csv")

| Facility (Supplier) | Facility ID (Unnamed: 2) | Reception Desk Count (2025 Q4) | Staff Count | Fixed Cost |
|---------------------|--------------------------|-------------------------------|-------------|------------|
| Facility_1          | MOUNT AYR                | 3                             | 12          | 96.58      |
| Facility_2          | WAUKEE                   | 3                             | 12          | 94.06      |
| Facility_3          | WAVERLY                  | 2                             | 20          | 94.37      |
| Facility_4          | PELLA                    | 2                             | 8           | 82.88      |
| Facility_5          | DES MOINES               | 3                             | 50          | 94.96      |

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

#### Each row is a supplier (facility), each column is a store (customer). The cost is per unit shipped.

| Facility (Supplier) | Facility ID (Unnamed: 2) | CLARINDA (Customer_1) | FORT MADISON (Customer_2) | SIOUX CITY (Customer_3) | TOLEDO (Customer_4) | BANCROFT (Customer_5) |
|---------------------|--------------------------|-----------------------|---------------------------|------------------------|---------------------|-----------------------|
| Facility_1          | MOUNT AYR                | 694.68                | 17.48                     | 20.07                  | 199.02              | 1685.53               |
| Facility_2          | WAUKEE                   | 15.13                 | 1.50                      | 1.43                   | 27.88               | 90.69                 |
| Facility_3          | WAVERLY                  | 2.34                  | 349.34                    | 246.60                 | 41.30               | 78.73                 |
| Facility_4          | PELLA                    | 1181.60               | 1458.53                   | 1646.36                | 1924.55             | 38.93                 |
| Facility_5          | DES MOINES               | 1030.80               | 43.48                     | 932.43                 | 55.39               | 103.84                |

- **Note:** The mapping between customer names and IDs is inferred from the order in the demand list and the columns in the cost matrix, as no explicit mapping is provided. The axis orientation is preserved as:  
  - **Rows:** Facilities/Suppliers (MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES)  
  - **Columns:** Customers/Stores (CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT)

---

### 4. Summary of Preserved Data Structure

- **Facility IDs:** MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES
- **Customer IDs:** CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT
- **Fixed Costs:** As above, per facility
- **Demand:** As above, per customer
- **Transportation Costs:** As above, per facility-customer pair

---

**All data is retrieved and preserved as requested, with no narrowing or omission.**