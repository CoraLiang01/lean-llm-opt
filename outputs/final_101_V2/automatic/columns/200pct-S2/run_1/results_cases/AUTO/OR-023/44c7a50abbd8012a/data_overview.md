Below is the complete retrieval of all relevant data from the provided context, preserving all facility (supplier) and customer (store) identifiers, fixed costs, demand, and the transportation cost matrix. All source-row positions and axis orientations are retained as in the original data.

---

### 1. Fixed Costs and Facility (Supplier) Data (`fixed_cost.csv`)

| Facility ID   | Facility Name | Fixed Cost | Staff Count | Reception Desk Count | Staff Training Hours | Equipment Audit Count |
|---------------|--------------|------------|-------------|---------------------|---------------------|----------------------|
| Facility_1    | MOUNT AYR    | 96.58      | 12          | 3                   | 8                   | 1                    |
| Facility_2    | WAUKEE       | 94.06      | 12          | 3                   | 36                  | 1                    |
| Facility_3    | WAVERLY      | 94.37      | 20          | 2                   | 24                  | 3                    |
| Facility_4    | PELLA        | 82.88      | 8           | 2                   | 12                  | 2                    |
| Facility_5    | DES MOINES   | 94.96      | 50          | 3                   | 12                  | 3                    |

---

### 2. Customer (Store) Demand Data (`demand.csv`)

| Customer ID   | Demand |
|---------------|--------|
| Customer_1    | 2397   |
| Customer_2    | 1889   |
| Customer_3    | 2518   |
| Customer_4    | 3218   |
| Customer_5    | 1813   |

---

### 3. Transportation Cost Matrix (`transportation_costs.csv`)

#### Axis: Rows = Facilities (Suppliers), Columns = Customers (Stores)

| Facility \ Customer | Customer_1 | Customer_2 | Customer_3 | Customer_4 | Customer_5 |
|---------------------|------------|------------|------------|------------|------------|
| MOUNT AYR           | 694.68     | 17.48      | 20.07      | 199.02     | 1685.53    |
| WAUKEE              | 15.13      | 1.50       | 1.43       | 27.88      | 90.69      |
| WAVERLY             | 2.34       | 349.34     | 246.60     | 41.30      | 78.73      |
| PELLA               | 1181.60    | 1458.53    | 1646.36    | 1924.55    | 38.93      |
| DES MOINES          | 1030.80    | 43.48      | 932.43     | 55.39      | 103.84     |

- **Source-row positions**: Each row corresponds to a facility (supplier) in the order listed above.
- **Source-column positions**: Each column corresponds to a customer (store) in the order listed above.

---

### 4. Capacity Data

- **No explicit capacity values** are present in the provided data for any facility. Capacity is unresolved evidence (not zero).

---

### 5. Summary of Preserved Identifiers

- **Facility IDs**: MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES
- **Customer IDs**: Customer_1, Customer_2, Customer_3, Customer_4, Customer_5
- **Fixed Costs**: As above, matched to each facility.
- **Demand**: As above, matched to each customer.
- **Transportation Cost Matrix**: As above, with explicit facility-to-customer mapping.

---

**No data has been transposed, truncated, padded, zero-filled, or inferred beyond the provided context. All axes and identifiers are preserved as in the source.**