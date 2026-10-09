Below is the complete retrieval of all relevant data from the provided context, preserving all facility (supplier) and customer (store) IDs, fixed costs, demands, and the transportation cost matrix. All identifiers and values are retained as in the source, with explicit mapping of facilities to customers and cost-matrix axes.

---

### 1. Fixed Costs and Facility Information (from "fixed_cost.csv")

| Facility ID   | Facility Name | Fixed Cost | Staff Count | Reception Desk Count | Staff Training Hours | Equipment Audit Count |
|---------------|--------------|------------|-------------|---------------------|---------------------|----------------------|
| MOUNT AYR     | MOUNT AYR    | 96.58      | 12          | 3                   | 8                   | 1                    |
| WAUKEE        | WAUKEE       | 94.06      | 12          | 3                   | 36                  | 1                    |
| WAVERLY       | WAVERLY      | 94.37      | 20          | 2                   | 24                  | 3                    |
| PELLA         | PELLA        | 82.88      | 8           | 2                   | 12                  | 2                    |
| DES MOINES    | DES MOINES   | 94.96      | 50          | 3                   | 12                  | 3                    |

---

### 2. Customer (Store) Demand (from "demand.csv")

| Customer ID   | Demand |
|---------------|--------|
| Customer_1    | 2397   |
| Customer_2    | 1889   |
| Customer_3    | 2518   |
| Customer_4    | 3218   |
| Customer_5    | 1813   |

---

### 3. Transportation Costs (from "transportation_costs.csv")

#### Cost per unit from each facility (row) to each customer (column):

| Facility \ Customer | Customer_1 | Customer_2 | Customer_3 | Customer_4 | Customer_5 |
|---------------------|------------|------------|------------|------------|------------|
| MOUNT AYR           | 694.68     | 17.48      | 20.07      | 199.02     | 1685.53    |
| WAUKEE              | 15.13      | 1.50       | 1.43       | 27.88      | 90.69      |
| WAVERLY             | 2.34       | 349.34     | 246.60     | 41.30      | 78.73      |
| PELLA               | 1181.60    | 1458.53    | 1646.36    | 1924.55    | 38.93      |
| DES MOINES          | 1030.80    | 43.48      | 932.43     | 55.39      | 103.84     |

- **Row orientation:** Each row is a facility (supplier).
- **Column orientation:** Each column is a customer (store).
- **All values are per-unit transportation costs.**

---

### 4. Capacity

- **No explicit capacity values are present in the provided data.** If capacity is required for modeling, it is unresolved based on current evidence.

---

### 5. Source Row Positions

- Facility and customer IDs are preserved as in the source.
- Matrix orientation is: rows = facilities, columns = customers.

---

**Summary:**  
All data from fixed_cost.csv, transportation_costs.csv, and demand.csv is retrieved and preserved in its original structure, with all identifiers and values intact. No data is omitted, transposed, or inferred beyond the provided context.