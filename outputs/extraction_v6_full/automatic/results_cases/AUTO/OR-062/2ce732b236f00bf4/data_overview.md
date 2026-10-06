Here is the complete retrieval of all data from the provided context, preserving all identifiers and values from fixed_cost.csv, transportation_costs.csv, and demand.csv:

---

### Demand Data (demand.csv)
| Customer    | Demand |
|-------------|--------|
| Customer_1  | 2397   |
| Customer_2  | 1889   |
| Customer_3  | 2518   |
| Customer_4  | 3218   |
| Customer_5  | 1813   |

---

### Fixed Cost Data (fixed_cost.csv)
| Facility      | Fixed_Cost |
|---------------|------------|
| MOUNT AYR     | 96.58      |
| WAUKEE        | 94.06      |
| WAVERLY       | 94.37      |
| PELLA         | 82.88      |
| DES MOINES    | 94.96      |

---

### Transportation Cost Data (transportation_costs.csv)
#### (Rows: Facilities, Columns: Customers)
| Facility      | Customer_1 (CLARINDA) | Customer_2 (FORT MADISON) | Customer_3 (SIOUX CITY) | Customer_4 (TOLEDO) | Customer_5 (BANCROFT) |
|---------------|-----------------------|---------------------------|-------------------------|---------------------|-----------------------|
| MOUNT AYR     | 694.68                | 17.48                     | 20.07                   | 199.02              | 1685.53               |
| WAUKEE        | 15.13                 | 1.50                      | 1.43                    | 27.88               | 90.69                 |
| WAVERLY       | 2.34                  | 349.34                    | 246.60                  | 41.30               | 78.73                 |
| PELLA         | 1181.60               | 1458.53                   | 1646.36                 | 1924.55             | 38.93                 |
| DES MOINES    | 1030.80               | 43.48                     | 932.43                  | 55.39               | 103.84                |

---

#### Source-row positions and axis mapping:
- Facilities (rows): MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES
- Customers (columns): Customer_1 (CLARINDA), Customer_2 (FORT MADISON), Customer_3 (SIOUX CITY), Customer_4 (TOLEDO), Customer_5 (BANCROFT)
- Fixed costs are matched to the same facility IDs as the transportation cost matrix rows.
- Demand is matched to the customer IDs as the transportation cost matrix columns.

---

**No capacity data is present in the provided context.**

All data is preserved in its original orientation and shape, with explicit mapping of facility and customer IDs.