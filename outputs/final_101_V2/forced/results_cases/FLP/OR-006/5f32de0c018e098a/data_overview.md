Here is the complete data retrieved from the three files, preserving all identifiers, values, and matrix orientations:

---

**customer_demand.csv**

| customer | demand |
|----------|--------|
| C1       | 45     |
| C2       | 23     |
| C3       | 94     |
| C4       | 92     |
| C5       | 57     |
| C6       | 52     |
| C7       | 23     |
| C8       | 99     |
| C9       | 99     |
| C10      | 77     |

---

**supply_capacity.csv**

| Unnamed: 0 | supply_capacity |
|------------|----------------|
| S1         | 127            |
| S2         | 236            |
| S3         | 168            |
| S4         | 115            |
| S5         | 280            |
| S6         | 179            |
| S7         | 135            |
| S8         | 263            |
| S9         | 283            |
| S10        | 476            |

---

**transportation_costs.csv**  
(*Each row is a warehouse, each column is a customer. Values are per-unit transportation costs.*)

| Unnamed: 0 |   C1   |   C2   |   C3   |   C4   |   C5   |   C6   |   C7   |   C8   |   C9   |   C10  |
|------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| S1         | 2077.06| 0.00   | 54.34  | 0.00   | 0.00   | 36.17  | 0.00   | 0.00   | 169.33 | 0.00   |
| S2         | 2077.06| 0.00   | 1141.04| 0.00   | 0.00   | 651.11 | 0.00   | 0.00   | 8.06   | 0.00   |
| S3         | 79.92  | 474.25 | 1477.07| 22.58  | 474.25 | 41.11  | 474.25 | 474.25 | 624.16 | 474.25 |
| S4         | 1659.34| 57.21  | 186.15 | 1201.31| 1029.70| 41.82  | 57.21  | 1201.31| 884.56 | 1029.70|
| S5         | 1297.26| 77.77  | 24.27  | 1399.79| 77.77  | 53.91  | 1399.79| 77.77  | 1255.12| 1399.79|
| S6         | 1998.91| 985.32 | 2.85   | 1149.54| 985.32 | 730.69 | 54.74  | 985.32 | 46.80  | 1149.54|
| S7         | 1780.34| 0.00   | 1141.04| 0.00   | 0.00   | 36.17  | 0.00   | 0.00   | 8.06   | 0.00   |
| S8         | 75.41  | 1338.20| 21.39  | 74.34  | 74.34  | 937.35 | 1338.20| 1338.20| 1392.12| 1338.20|
| S9         | 98.91  | 0.00   | 978.03 | 0.00   | 0.00   | 651.11 | 0.00   | 0.00   | 169.33 | 0.00   |
| S10        | 2077.06| 0.00   | 54.34  | 0.00   | 0.00   | 36.17  | 0.00   | 0.00   | 145.14 | 0.00   |

---

**Preserved structure:**
- Facility IDs: S1–S10 (rows in supply_capacity and transportation_costs)
- Customer IDs: C1–C10 (columns in customer_demand and transportation_costs)
- Each cost-matrix row and column matches the facility and customer IDs, with no transposition or inferred axes.
- All values and identifiers are retained as in the source data.