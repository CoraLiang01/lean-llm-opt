Below is the complete retrieval of all relevant data from the provided context, preserving all facility IDs, customer IDs, fixed costs, and the full cost matrix, with explicit source-row positions and axis orientation. No data is omitted, transposed, or inferred beyond the original context.

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

#### Axis and Shape Preservation:
- Facilities (S1, S2) are the rows in the cost matrix and fixed cost table.
- Customers (C1, C2) are the columns in the demand and cost matrix.
- No capacity data is present; capacity is unresolved.
- All IDs and values are preserved as in the source.

---

**Summary Table for Modeling:**

| Facility | Fixed Cost | C1 Cost | C2 Cost |
|----------|-----------|---------|---------|
| S1       | 105.97    | 2358.39 | 1492.08 |
| S2       | 85.31     | 0.07    | 52.32   |

| Customer | Demand |
|----------|--------|
| C1       | 144    |
| C2       | 216    |

---

**All data from fixed_cost.csv, transportation_costs.csv, and demand.csv has been retrieved and preserved with explicit identifiers and source orientation.**