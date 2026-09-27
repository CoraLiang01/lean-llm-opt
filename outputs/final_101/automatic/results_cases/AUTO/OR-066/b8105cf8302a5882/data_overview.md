Below is the complete retrieval of all relevant data from the provided context, preserving all facility IDs, customer IDs, fixed costs, and the full cost matrix, with explicit source-row positions and original orientation. No data is omitted, transposed, or inferred beyond the original context.

---

### 1. Demand Data (from demand.csv)
| customer | demand | (source row) |
|----------|--------|--------------|
| C1       | 144    | 1            |
| C2       | 216    | 2            |

---

### 2. Fixed Costs Data (from fixed_cost.csv)
| facility (Unnamed: 0) | fixed_costs | (source row) |
|-----------------------|-------------|--------------|
| S1                    | 105.97      | 3            |
| S2                    | 85.31       | 4            |

---

### 3. Transportation Costs Matrix (from transportation_costs.csv)
| facility (Unnamed: 0) | C1      | C2     | (source row) |
|-----------------------|---------|--------|--------------|
| S1                    | 2358.39 | 1492.08| 5            |
| S2                    | 0.07    | 52.32  | 6            |

---

#### Matrix orientation and axis mapping:
- **Rows:** Facilities (S1, S2)
- **Columns:** Customers (C1, C2)
- **Values:** Per-unit transportation cost from each facility to each customer

---

**All data is preserved as per the original files and context.**