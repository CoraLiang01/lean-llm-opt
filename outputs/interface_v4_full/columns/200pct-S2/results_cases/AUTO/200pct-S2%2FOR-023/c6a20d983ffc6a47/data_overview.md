Below is the complete retrieval of all relevant data from the provided context, preserving all facility (supplier) and customer (store) IDs, fixed costs, demands, and the transportation cost matrix. All identifiers and values are retained as in the source, with explicit mapping of facilities to customers and cost-matrix axes. No transposition, truncation, or inference is performed.

---

### Facilities (Suppliers) — from `fixed_cost.csv`
| Facility ID   | Facility Name | Fixed Cost | Staff Count | Staff Training Hours | Reception Desk Count | Equipment Audit Count |
|---------------|--------------|------------|-------------|---------------------|----------------------|----------------------|
| 1             | MOUNT AYR    | 96.58      | 12          | 8                   | 3                    | 1                    |
| 2             | WAUKEE       | 94.06      | 12          | 36                  | 3                    | 1                    |
| 3             | WAVERLY      | 94.37      | 20          | 24                  | 2                    | 3                    |
| 4             | PELLA        | 82.88      | 8           | 12                  | 2                    | 2                    |
| 5             | DES MOINES   | 94.96      | 50          | 12                  | 3                    | 3                    |

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
#### Cost per unit from each facility (row) to each customer (column):

| Facility \ Customer | Customer_1 | Customer_2 | Customer_3 | Customer_4 | Customer_5 |
|---------------------|------------|------------|------------|------------|------------|
| MOUNT AYR           | 694.68     | 17.48      | 20.07      | 199.02     | 1685.53    |
| WAUKEE              | 15.13      | 1.50       | 1.43       | 27.88      | 90.69      |
| WAVERLY             | 2.34       | 349.34     | 246.60     | 41.30      | 78.73      |
| PELLA               | 1181.60    | 1458.53    | 1646.36    | 1924.55    | 38.93      |
| DES MOINES          | 1030.80    | 43.48      | 932.43     | 55.39      | 103.84     |

- **Source orientation:** Rows = Facilities (suppliers), Columns = Customers (stores).
- **All values are as provided in the source.**

---

### Summary of Preserved Data Structure

- **Facility IDs:** 1 (MOUNT AYR), 2 (WAUKEE), 3 (WAVERLY), 4 (PELLA), 5 (DES MOINES)
- **Customer IDs:** Customer_1, Customer_2, Customer_3, Customer_4, Customer_5
- **Fixed Costs:** As above, matched to facility IDs.
- **Demands:** As above, matched to customer IDs.
- **Transportation Cost Matrix:** 5x5, as above, with explicit facility-to-customer mapping.

---

**No capacity data is present in the provided context.** All other required data for the two-dimensional shipment decision (facility-to-customer) is included as requested.