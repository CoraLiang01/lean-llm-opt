**Retrieved Data for Model Formulation**

---

### 1. Demand Data (`demand.csv`)

**Source order preserved; each row is a customer (branch) with demand units for the current period:**

| customer_id | demand_units | three_periods_ago_demand_units | two_periods_ago_demand_units | demand_previous_period | four_periods_ago_demand_units |
|-------------|--------------|-------------------------------|-----------------------------|-----------------------|-------------------------------|
| C1          | 143          | 157                           | 133                         | 149                   | 151                           |
| C2          | 6            | 7                             | 5                           | 7                     | 5                             |
| C3          | 10           | 8                             | 11                          | 9                     | 12                            |
| C4          | 25           | 27                            | 26                          | 26                    | 29                            |
| C5          | 3            | 4                             | 4                           | 4                     | 4                             |

---

### 2. Fixed Cost Data (`fixed_cost.csv`)

**Source order preserved; each row is a supplier (facility) with fixed opening cost for the current period:**

| facility_id | fixed_opening_cost | three_periods_ago_fixed_opening_cost | two_periods_ago_fixed_opening_cost | fixed_opening_cost_previous_period | four_periods_ago_fixed_opening_cost |
|-------------|-------------------|--------------------------------------|------------------------------------|------------------------------------|-------------------------------------|
| S1          | 97.65             | 115.070760                           | 109.260585                         | 89.720820                          | 86.361660                          |
| S2          | 99.76             | 97.325856                            | 83.040224                          | 115.033256                         | 106.703296                         |
| S3          | 100.76            | 91.510232                            | 87.25816                           | 99.117612                          | 97.021804                          |
| S4          | 105.32            | 109.069392                           | 126.015380                         | 86.867936                          | 114.472308                         |
| S5          | 98.88             | 88.458048                            | 90.346656                          | 95.340096                          | 85.086240                          |

---

### 3. Transportation Cost Data (`transportation_costs.csv`)

**Source order preserved; each row is a supplier (facility), each column is a customer (branch):**

| facility_id | transportation_cost_to_C1 | transportation_cost_to_C2 | transportation_cost_to_C3 | transportation_cost_to_C4 | transportation_cost_to_C5 |
|-------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| S1          | 150.74                   | 0.02                     | 49.13                    | 2080.15                  | 426.4                    |
| S2          | 233.05                   | 97.73                    | 49.84                    | 1982.39                  | 23.96                    |
| S3          | 55.68                    | 935.61                   | 4.03                     | 73.09                    | 525.32                   |
| S4          | 1483.82                  | 1801.08                  | 112.16                   | 816.05                   | 107.01                   |
| S5          | 1119.47                  | 884.31                   | 0.08                     | 1544.95                  | 543.67                   |

---

**All identifiers, values, and source order are preserved. No capacity data is present; capacity is unresolved. No data is omitted, transposed, or inferred.**

---

**Summary of Model Axes:**

- **Facilities (Suppliers):** S1, S2, S3, S4, S5
- **Customers (Branches):** C1, C2, C3, C4, C5
- **Fixed Costs:** As above, by facility
- **Demand:** As above, by customer
- **Transportation Cost Matrix:** As above, facility-to-customer

**Ready for two-dimensional shipment decision modeling.**