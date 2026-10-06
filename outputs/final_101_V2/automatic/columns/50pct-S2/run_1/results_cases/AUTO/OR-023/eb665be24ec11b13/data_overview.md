**Retrieved Data:**

---

### 1. Fixed Cost Data (`fixed_cost.csv`)

| Facility ID   | Facility Name | Facility Staff Count | Fixed Cost |
|---------------|--------------|---------------------|------------|
| 1             | MOUNT AYR    | 12                  | 96.58      |
| 2             | WAUKEE       | 12                  | 94.06      |
| 3             | WAVERLY      | 20                  | 94.37      |
| 4             | PELLA        | 8                   | 82.88      |
| 5             | DES MOINES   | 50                  | 94.96      |

_Source: Rows 1–5 of fixed_cost.csv_

---

### 2. Transportation Cost Matrix (`transportation_costs.csv`)

| Facility (Row) | Customer (Column) | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|----------------|-------------------|----------|--------------|------------|--------|----------|
| MOUNT AYR      | -                 | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE         | -                 | 15.13    | 1.5          | 1.43       | 27.88  | 90.69    |
| WAVERLY        | -                 | 2.34     | 349.34       | 246.6      | 41.3   | 78.73    |
| PELLA          | -                 | 1181.6   | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES     | -                 | 1030.8   | 43.48        | 932.43     | 55.39  | 103.84   |

_Source: Rows 1–5 of transportation_costs.csv; matrix orientation is Facility (row) to Customer (column)_

---

### 3. Demand Data (`demand.csv`)

| Customer ID | Customer Name | Demand |
|-------------|--------------|--------|
| 1           | Customer_1   | 2397   |
| 2           | Customer_2   | 1889   |
| 3           | Customer_3   | 2518   |
| 4           | Customer_4   | 3218   |
| 5           | Customer_5   | 1813   |

_Source: Rows 1–5 of demand.csv_

---

**Preserved Identifiers and Source Orientation:**
- Facility IDs: MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES
- Customer IDs: Customer_1, Customer_2, Customer_3, Customer_4, Customer_5
- Fixed costs and staff counts are matched to facility IDs.
- Transportation cost matrix is Facility (row) × Customer (column), with explicit facility and customer names.
- Demand is per customer, with explicit customer IDs.

**No capacity data is present; capacity is unresolved.**

**No product axis is present; all data is for a single product as per the query.**

---

**All data from the specified files is retrieved and preserved as requested.**