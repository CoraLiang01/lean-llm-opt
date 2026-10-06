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

#### Axis and ID Preservation:
- Facilities: S1, S2 (as per fixed_costs and transportation_costs rows)
- Customers: C1, C2 (as per demand and transportation_costs columns)
- FixedCost and cost-matrix rows both use facility IDs S1, S2.
- Demand is per customer ID C1, C2.

#### No capacity data is present; capacity is unresolved.

---

**Summary Table (for clarity, not for modeling):**

| Facility | Fixed Cost | To C1 (cost) | To C2 (cost) |
|----------|------------|--------------|--------------|
| S1       | 105.97     | 2358.39      | 1492.08      |
| S2       | 85.31      | 0.07         | 52.32        |

| Customer | Demand |
|----------|--------|
| C1       | 144    |
| C2       | 216    |

---

**All data is retrieved as requested, with explicit preservation of all identifiers and source orientation.**