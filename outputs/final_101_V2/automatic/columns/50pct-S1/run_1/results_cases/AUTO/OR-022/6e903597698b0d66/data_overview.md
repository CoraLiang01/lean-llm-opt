Below is all the data required to formulate the described facility location and transportation model, preserving source order, exact identifiers, and values. All facility IDs, customer IDs, fixed costs, demand, and the full cost matrix are included, with explicit axis labels and source-row positions retained.

---

### 1. Customer Demand Data (from 'demand.csv')
| Source Row | customer_id | archive_revision_number | demand_units |
|------------|-------------|------------------------|--------------|
| 1          | C1          | 6                      | 143          |
| 2          | C2          | 5                      | 6            |
| 3          | C3          | 3                      | 10           |
| 4          | C4          | 1                      | 25           |
| 5          | C5          | 1                      | 3            |

---

### 2. Facility Fixed Opening Cost Data (from 'fixed_cost.csv')
| Source Row | facility_id | archive_revision_number | fixed_opening_cost |
|------------|-------------|------------------------|--------------------|
| 6          | S1          | 3                      | 97.65              |
| 7          | S2          | 5                      | 99.76              |
| 8          | S3          | 1                      | 100.76             |
| 9          | S4          | 2                      | 105.32             |
| 10         | S5          | 5                      | 98.88              |

---

### 3. Transportation Cost Matrix (from 'transportation_costs.csv')
#### Each row corresponds to a facility (supplier), each column to a customer (branch). All costs are per unit shipped.

| Source Row | facility_id | archive_revision_number | transportation_cost_to_C1 | transportation_cost_to_C2 | transportation_cost_to_C3 | transportation_cost_to_C4 | transportation_cost_to_C5 | document_page_count | record_display_theme |
|------------|-------------|------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|---------------------|---------------------|
| 11         | S1          | 2                      | 150.74                   | 0.02                     | 49.13                    | 2080.15                  | 426.4                    | 16                  | Azure               |
| 12         | S2          | 2                      | 233.05                   | 97.73                    | 49.84                    | 1982.39                  | 23.96                    | 4                   | Slate               |
| 13         | S3          | 2                      | 55.68                    | 935.61                   | 4.03                     | 73.09                    | 525.32                   | 12                  | Olive               |
| 14         | S4          | 4                      | 1483.82                  | 1801.08                  | 112.16                   | 816.05                   | 107.01                   | 8                   | Slate               |
| 15         | S5          | 5                      | 1119.47                  | 884.31                   | 0.08                     | 1544.95                  | 543.67                   | 2                   | Olive               |

---

### 4. Explicit Axis and Shape Information

- **Facilities (Suppliers):** S1, S2, S3, S4, S5
- **Customers (Branches):** C1, C2, C3, C4, C5
- **Fixed Opening Cost:** Each facility S_i has a fixed_opening_cost as above.
- **Demand:** Each customer C_j has a demand_units as above.
- **Transportation Cost Matrix:** Rows = facilities (S1–S5), Columns = customers (C1–C5), values as above.
- **No capacity data is present for facilities.** (Capacity is unresolved evidence, not zero.)

---

### 5. Decision Variables

- **y_i:** Binary, 1 if facility S_i is open, 0 otherwise.
- **x_{ij}:** Quantity shipped from facility S_i to customer C_j.

---

**All data is preserved in original source order and with exact identifiers and values. No transposition, truncation, or inference beyond the provided data.**