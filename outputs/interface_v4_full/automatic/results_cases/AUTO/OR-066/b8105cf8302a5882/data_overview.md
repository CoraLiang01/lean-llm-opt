Below is the complete retrieval of all relevant data from the provided context, preserving all identifiers, values, and source-row positions. No data is omitted or inferred beyond the original context.

---

### 1. fixed_cost.csv

| Facility ID | FixedCost | Source Row Position |
|-------------|-----------|--------------------|
| S1          | 105.97    | Row 1              |
| S2          | 85.31     | Row 2              |

---

### 2. transportation_costs.csv

| Facility ID | Customer ID | Transportation Cost | Source Row Position | Matrix Orientation |
|-------------|-------------|--------------------|--------------------|-------------------|
| S1          | C1          | 2358.39            | Row S1             | Facility-to-Customer |
| S1          | C2          | 1492.08            | Row S1             | Facility-to-Customer |
| S2          | C1          | 0.07               | Row S2             | Facility-to-Customer |
| S2          | C2          | 52.32              | Row S2             | Facility-to-Customer |

---

### 3. demand.csv

| Customer ID | Demand | Source Row Position |
|-------------|--------|--------------------|
| C1          | 144    | Row 1              |
| C2          | 216    | Row 2              |

---

**Notes:**
- All facility IDs (S1, S2) and customer IDs (C1, C2) are preserved as in the source.
- FixedCost and transportation cost axes are explicitly matched to their facility and customer IDs.
- No capacity data is present; absence is unresolved evidence, not zero.
- The data is not transposed, truncated, padded, or inferred beyond the provided context.
- The matrix orientation is facility-to-customer, as per the source.

---

**Summary Table for Modeling:**

| Facility | FixedCost | To Customer | Transportation Cost | Customer Demand |
|----------|-----------|-------------|--------------------|----------------|
| S1       | 105.97    | C1          | 2358.39            | 144            |
| S1       | 105.97    | C2          | 1492.08            | 216            |
| S2       | 85.31     | C1          | 0.07               | 144            |
| S2       | 85.31     | C2          | 52.32              | 216            |

All data is preserved and ready for use in a two-dimensional shipment decision model.