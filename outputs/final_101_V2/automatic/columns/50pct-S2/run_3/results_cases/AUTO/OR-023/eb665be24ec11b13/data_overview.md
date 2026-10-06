**Retrieved Data**

---

### 1. Fixed Cost Data (`fixed_cost.csv`)
| Facility ID   | Facility Name | Facility Staff Count | Fixed Cost |
|---------------|--------------|---------------------|------------|
| F1            | MOUNT AYR    | 12                  | 96.58      |
| F2            | WAUKEE       | 12                  | 94.06      |
| F3            | WAVERLY      | 20                  | 94.37      |
| F4            | PELLA        | 8                   | 82.88      |
| F5            | DES MOINES   | 50                  | 94.96      |

_Source: Rows 1–5 of fixed_cost.csv_

---

### 2. Transportation Cost Matrix (`transportation_costs.csv`)
| Facility (Row) | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|----------------|----------|--------------|------------|--------|----------|
| MOUNT AYR      | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE         | 15.13    | 1.50         | 1.43       | 27.88  | 90.69    |
| WAVERLY        | 2.34     | 349.34       | 246.60     | 41.30  | 78.73    |
| PELLA          | 1181.60  | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES     | 1030.80  | 43.48        | 932.43     | 55.39  | 103.84   |

_Source: Rows 1–5 of transportation_costs.csv; each row is a facility, each column is a customer/store._

---

### 3. Demand Data (`demand.csv`)
| Customer ID | Customer Name | Demand |
|-------------|--------------|--------|
| S1          | Customer_1   | 2397   |
| S2          | Customer_2   | 1889   |
| S3          | Customer_3   | 2518   |
| S4          | Customer_4   | 3218   |
| S5          | Customer_5   | 1813   |

_Source: Rows 1–5 of demand.csv_

---

**Preserved Identifiers and Source Orientation:**
- Facility IDs: F1 (MOUNT AYR), F2 (WAUKEE), F3 (WAVERLY), F4 (PELLA), F5 (DES MOINES)
- Customer IDs: S1 (Customer_1), S2 (Customer_2), S3 (Customer_3), S4 (Customer_4), S5 (Customer_5)
- Fixed costs and staff counts are matched to facility IDs.
- Transportation cost matrix: rows = facilities, columns = customers (CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT).
- Demand vector: each customer/store’s demand.

**No capacity data is present; capacity is unresolved.**

**No product axis is present; all data is for a single product.**

---

**All data is retrieved as requested, with all identifiers and source-row positions preserved.**