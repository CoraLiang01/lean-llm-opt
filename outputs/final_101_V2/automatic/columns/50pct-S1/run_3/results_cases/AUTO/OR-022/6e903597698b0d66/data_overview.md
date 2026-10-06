**Retrieved Data for Model Formulation (Preserving Source Order, Identifiers, and Values):**

---

**1. Demand Data (from 'demand.csv'):**

| Source Row | customer_id | archive_revision_number | demand_units |
|------------|-------------|------------------------|-------------|
| 1          | C1          | 6                      | 143         |
| 2          | C2          | 5                      | 6           |
| 3          | C3          | 3                      | 10          |
| 4          | C4          | 1                      | 25          |
| 5          | C5          | 1                      | 3           |

---

**2. Fixed Opening Cost Data (from 'fixed_cost.csv'):**

| Source Row | facility_id | archive_revision_number | fixed_opening_cost |
|------------|-------------|------------------------|--------------------|
| 6          | S1          | 3                      | 97.65              |
| 7          | S2          | 5                      | 99.76              |
| 8          | S3          | 1                      | 100.76             |
| 9          | S4          | 2                      | 105.32             |
| 10         | S5          | 5                      | 98.88              |

---

**3. Transportation Cost Matrix (from 'transportation_costs.csv'):**

**Source Row 11: facility_id S1, archive_revision_number 2**
- transportation_cost_to_C1: 150.74
- transportation_cost_to_C2: 0.02
- transportation_cost_to_C3: 49.13
- transportation_cost_to_C4: 2080.15
- transportation_cost_to_C5: 426.4

**Source Row 12: facility_id S2, archive_revision_number 2**
- transportation_cost_to_C1: 233.05
- transportation_cost_to_C2: 97.73
- transportation_cost_to_C3: 49.84
- transportation_cost_to_C4: 1982.39
- transportation_cost_to_C5: 23.96

**Source Row 13: facility_id S3, archive_revision_number 2**
- transportation_cost_to_C1: 55.68
- transportation_cost_to_C2: 935.61
- transportation_cost_to_C3: 4.03
- transportation_cost_to_C4: 73.09
- transportation_cost_to_C5: 525.32

**Source Row 14: facility_id S4, archive_revision_number 4**
- transportation_cost_to_C1: 1483.82
- transportation_cost_to_C2: 1801.08
- transportation_cost_to_C3: 112.16
- transportation_cost_to_C4: 816.05
- transportation_cost_to_C5: 107.01

**Source Row 15: facility_id S5, archive_revision_number 5**
- transportation_cost_to_C1: 1119.47
- transportation_cost_to_C2: 884.31
- transportation_cost_to_C3: 0.08
- transportation_cost_to_C4: 1544.95
- transportation_cost_to_C5: 543.67

---

**Summary of Preserved Axes and Identifiers:**

- **Facilities (Suppliers):** S1, S2, S3, S4, S5
- **Customers (Branches):** C1, C2, C3, C4, C5
- **Fixed Opening Cost:** For each facility, as above, with matching facility_id.
- **Demand:** For each customer, as above, with matching customer_id.
- **Transportation Cost Matrix:** Each row is a facility (S1–S5), each column is a customer (C1–C5), with costs as above, preserving source orientation and shape.

**No capacity data is present; capacity is unresolved.**

**No product/commodity axis is present or implied.**

**All data is preserved in original source order and with explicit identifiers.**