Below is the complete retrieval of all relevant data from the provided context, preserving all facility (supplier) and customer (store) IDs, fixed costs, demands, and the transportation cost matrix. All identifiers and values are retained as in the source, with explicit mapping of facilities to customers and cost-matrix axes. No transposition, truncation, or inference is performed.

---

### Facilities (Suppliers) — from `fixed_cost.csv`
| Facility ID   | Facility Name | Fixed Cost | Staff Count | Staff Training Hours (2025 Q4) | Reception Desk Count | Equipment Audit Count (2025 Q4) |  
|---------------|--------------|------------|-------------|-------------------------------|---------------------|-------------------------------|  
| F1            | MOUNT AYR    | 96.58      | 12          | 8                             | 3                   | 1                             |  
| F2            | WAUKEE       | 94.06      | 12          | 36                            | 3                   | 1                             |  
| F3            | WAVERLY      | 94.37      | 20          | 24                            | 2                   | 3                             |  
| F4            | PELLA        | 82.88      | 8           | 12                            | 2                   | 2                             |  
| F5            | DES MOINES   | 94.96      | 50          | 12                            | 3                   | 3                             |  

---

### Customers (Stores) — from `demand.csv`
| Customer ID   | Demand |  
|---------------|--------|  
| Customer_1    | 2397   |  
| Customer_2    | 1889   |  
| Customer_3    | 2518   |  
| Customer_4    | 3218   |  
| Customer_5    | 1813   |  

---

### Transportation Costs — from `transportation_costs.csv`
Each entry is the cost per unit to ship from Facility F_i to Customer S_j.

| Facility \ Customer | Customer_1 | Customer_2 | Customer_3 | Customer_4 | Customer_5 |  
|---------------------|------------|------------|------------|------------|------------|  
| MOUNT AYR           | 694.68     | 17.48      | 20.07      | 199.02     | 1685.53    |  
| WAUKEE              | 15.13      | 1.50       | 1.43       | 27.88      | 90.69      |  
| WAVERLY             | 2.34       | 349.34     | 246.60     | 41.30      | 78.73      |  
| PELLA               | 1181.60    | 1458.53    | 1646.36    | 1924.55    | 38.93      |  
| DES MOINES          | 1030.80    | 43.48      | 932.43     | 55.39      | 103.84     |  

- The rows correspond to facilities (suppliers):  
  - Row 1: MOUNT AYR  
  - Row 2: WAUKEE  
  - Row 3: WAVERLY  
  - Row 4: PELLA  
  - Row 5: DES MOINES  

- The columns correspond to customers (stores):  
  - Column 1: Customer_1  
  - Column 2: Customer_2  
  - Column 3: Customer_3  
  - Column 4: Customer_4  
  - Column 5: Customer_5  

---

**All data is preserved as in the source, with explicit facility and customer IDs, fixed costs, and the full cost matrix. No capacity data is present; thus, capacity is unresolved.**