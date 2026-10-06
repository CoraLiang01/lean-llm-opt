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

**3. Transportation Cost Data (from 'transportation_costs.csv'):**

| Source Row | facility_id | archive_revision_number | transportation_cost_to_C1 | transportation_cost_to_C2 | transportation_cost_to_C3 | transportation_cost_to_C4 | transportation_cost_to_C5 | document_page_count | record_display_theme |
|------------|-------------|------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|---------------------|---------------------|
| 11         | S1          | 2                      | 150.74                   | 0.02                     | 49.13                    | 2080.15                  | 426.4                    | 16                  | Azure               |
| 12         | S2          | 2                      | 233.05                   | 97.73                    | 49.84                    | 1982.39                  | 23.96                    | 4                   | Slate               |
| 13         | S3          | 2                      | 55.68                    | 935.61                   | 4.03                     | 73.09                    | 525.32                   | 12                  | Olive               |
| 14         | S4          | 4                      | 1483.82                  | 1801.08                  | 112.16                   | 816.05                   | 107.01                   | 8                   | Slate               |
| 15         | S5          | 5                      | 1119.47                  | 884.31                   | 0.08                     | 1544.95                  | 543.67                   | 2                   | Olive               |

---

**Summary of Preserved Axes and Identifiers:**

- **Facilities (Suppliers):** S1, S2, S3, S4, S5
- **Customers (Branches):** C1, C2, C3, C4, C5
- **Fixed Opening Cost:** For each facility, as above.
- **Demand:** For each customer, as above.
- **Transportation Cost Matrix:** Each facility (row) to each customer (column), as above.

**No capacity data is present; capacity is unresolved.**

**All data is preserved in original source order and with explicit identifiers.**