Below is the complete retrieval of all relevant data from the provided context, preserving all facility IDs, customer IDs, FixedCost, Capacity (if present), Demand, and the cost-matrix axis, with explicit source-row positions and no transposition, truncation, or inference beyond the original data.

---

### 1. Demand Data (`demand.csv`)

| Customer ID | Demand | Source Row Position |
|-------------|--------|--------------------|
| C1          | 1083   | 1                  |
| C2          | 776    | 2                  |
| C3          | 16214  | 3                  |

---

### 2. Fixed Cost Data (`fixed_cost.csv`)

| Facility ID | FixedCost | Source Row Position |
|-------------|-----------|--------------------|
| S1          | 102.33    | 4                  |
| S2          | 94.92     | 5                  |
| S3          | 91.83     | 6                  |

---

### 3. Transportation Cost Matrix (`transportation_costs.csv`)

**Matrix shape:** Facilities (rows: S1, S2, S3) × Customers (columns: C1, C2, C3)  
**Source orientation:** Each row is a facility, each column is a customer.

| Facility ID | C1      | C2      | C3     | Source Row Position |
|-------------|---------|---------|--------|--------------------|
| S1          | 1506.22 | 70.9    | 8.44   | 7                  |
| S2          | 1732.65 | 1780.72 | 567.44 | 8                  |
| S3          | 115.66  | 100.76  | 64.68  | 9                  |

---

### 4. Capacity Data

**No explicit capacity values are present in the provided data.**  
(Absence of capacity is unresolved evidence, not zero.)

---

### 5. Summary Table

#### Facilities (Warehouses)
- S1: FixedCost = 102.33
- S2: FixedCost = 94.92
- S3: FixedCost = 91.83

#### Customers (Musicians/Bands)
- C1: Demand = 1083
- C2: Demand = 776
- C3: Demand = 16214

#### Transportation Costs (per unit, from facility to customer)

|        | C1      | C2      | C3     |
|--------|---------|---------|--------|
| **S1** | 1506.22 | 70.9    | 8.44   |
| **S2** | 1732.65 | 1780.72 | 567.44 |
| **S3** | 115.66  | 100.76  | 64.68  |

---

**All identifiers and values are preserved as in the source. No data has been transposed, truncated, or inferred beyond the original context.**