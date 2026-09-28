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

### Transportation Cost Matrix (transportation_costs.csv)
| Facility      | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|---------------|----------|--------------|------------|--------|----------|
| MOUNT AYR     | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE        | 15.13    | 1.5          | 1.43       | 27.88  | 90.69    |
| WAVERLY       | 2.34     | 349.34       | 246.6      | 41.3   | 78.73    |
| PELLA         | 1181.6   | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES    | 1030.8   | 43.48        | 932.43     | 55.39  | 103.84   |

---

#### Source-row positions and axis mapping:
- Facility IDs: MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES (rows in fixed_cost and transportation_costs)
- Customer IDs: CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT (columns in transportation_costs)
- Demand IDs: Customer_1, Customer_2, Customer_3, Customer_4, Customer_5 (rows in demand.csv)
- FixedCost and transportation cost matrix rows both use the same facility IDs.
- No explicit capacity data is present.

---

All data is preserved in its original orientation and shape, with all identifiers and values intact.