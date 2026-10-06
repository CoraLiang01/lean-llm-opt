**Retrieved Data for Facility Location Model (Superstore Chain):**

---

### 1. Customer Demand Data (`demand.csv`)
| Source Row | customer_id | demand_units |
|------------|-------------|--------------|
| 1          | C1          | 143          |
| 2          | C2          | 6            |
| 3          | C3          | 10           |
| 4          | C4          | 25           |
| 5          | C5          | 3            |

---

### 2. Facility Fixed Cost Data (`fixed_cost.csv`)
| Source Row | facility_id | fixed_opening_cost |
|------------|-------------|--------------------|
| 6          | S1          | 97.65              |
| 7          | S2          | 99.76              |
| 8          | S3          | 100.76             |
| 9          | S4          | 105.32             |
| 10         | S5          | 98.88              |

---

### 3. Facility Capacity Data (from staff count as only available proxy)
| Source Row | facility_id | facility_staff_count |
|------------|-------------|---------------------|
| 6          | S1          | 8                   |
| 7          | S2          | 12                  |
| 8          | S3          | 50                  |
| 9          | S4          | 8                   |
| 10         | S5          | 20                  |

---

### 4. Transportation Cost Matrix (`transportation_costs.csv`)
**Rows: Facilities (S1–S5), Columns: Customers (C1–C5). Values: Transportation cost per unit.**

| Source Row | facility_id | transportation_cost_to_C1 | transportation_cost_to_C2 | transportation_cost_to_C3 | transportation_cost_to_C4 | transportation_cost_to_C5 |
|------------|-------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| 11         | S1          | 150.74                   | 0.02                     | 49.13                    | 2080.15                  | 426.4                    |
| 12         | S2          | 233.05                   | 97.73                    | 49.84                    | 1982.39                  | 23.96                    |
| 13         | S3          | 55.68                    | 935.61                   | 4.03                     | 73.09                    | 525.32                   |
| 14         | S4          | 1483.82                  | 1801.08                  | 112.16                   | 816.05                   | 107.01                   |
| 15         | S5          | 1119.47                  | 884.31                   | 0.08                     | 1544.95                  | 543.67                   |

---

**All data is preserved in original source order, with explicit identifiers and values. No transposition, truncation, or inference beyond the provided evidence.**