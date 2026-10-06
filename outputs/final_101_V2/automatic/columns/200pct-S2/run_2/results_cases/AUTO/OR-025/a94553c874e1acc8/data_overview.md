Below is the complete retrieval of all relevant data from the provided context, preserving all identifiers, values, and source-row positions. The data is organized by file/source as requested:

---

### fixed_cost.csv

| Facility ID | FixedCost | Capacity (if present) | Source-row position |
|-------------|-----------|-----------------------|--------------------|
| S1          | 105.97    | (unresolved)          | Row 1              |
| S2          | 85.31     | (unresolved)          | Row 2              |

---

### transportation_costs.csv

| Facility ID | Customer ID | Transportation Cost | Source orientation (matrix row: facility, column: customer) | Source-row position |
|-------------|-------------|---------------------|-------------------------------------------------------------|--------------------|
| S1          | C1          | 2358.39             | S1 to C1                                                   | Row S1             |
| S1          | C2          | 1492.08             | S1 to C2                                                   | Row S1             |
| S2          | C1          | 0.07                | S2 to C1                                                   | Row S2             |
| S2          | C2          | 52.32               | S2 to C2                                                   | Row S2             |

---

### demand.csv

| Customer ID | Demand | Source-row position |
|-------------|--------|--------------------|
| C1          | 144    | Row C1             |
| C2          | 216    | Row C2             |

---

**Notes:**
- Facility IDs: S1, S2
- Customer IDs: C1, C2
- FixedCost and transportation costs are explicitly matched to their respective facility and customer IDs.
- Capacity is not specified in the provided data and is thus unresolved (not zero).
- The cost matrix is two-dimensional: facilities (rows) × customers (columns).
- No extra product/commodity axis is present or inferred.

This preserves all required data and structure for modeling the shipment decision as described in your query.