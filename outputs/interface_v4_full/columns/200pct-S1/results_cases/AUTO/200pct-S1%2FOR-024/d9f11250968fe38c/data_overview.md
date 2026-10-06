**Retrieved Data**

---

### 1. Demand Data (`demand.csv`)

| Customer | Demand | Source Row |
|----------|--------|------------|
| C1       | 1083   | 1          |
| C2       | 776    | 2          |
| C3       | 16214  | 3          |

---

### 2. Fixed Cost Data (`fixed_cost.csv`)

| Facility (Warehouse) | Fixed Cost | Source Row | Archive Batch | Document Page Count | Archive Revision | Record View Count |
|----------------------|------------|------------|---------------|--------------------|------------------|-------------------|
| S1                   | 102.33     | 4          | 301           | 8                  | 2                | 43                |
| S2                   | 94.92      | 5          | 301           | 6                  | 4                | 27                |
| S3                   | 91.83      | 6          | 305           | 8                  | 2                | 58                |

---

### 3. Transportation Cost Matrix (`transportation_costs.csv`)

**Matrix Shape:** Facilities (rows: S1, S2, S3) × Customers (columns: C1, C2, C3)  
**Source Orientation:** Each row is a facility, each column is a customer.

| Facility (Warehouse) | C1      | C2      | C3     | Source Row | Archive Batch | Document Template Family | Record Label Font | Archive Storage Medium | Record Display Theme | Archive Revision | Record View Count | Document Page Count |
|----------------------|---------|---------|--------|------------|---------------|------------------------|-------------------|-----------------------|---------------------|------------------|-------------------|--------------------|
| S1                   | 1506.22 | 70.9    | 8.44   | 7          | 301           | Standard               | Helvetica         | Paper                 | Olive               | 4                | 76                | 6                  |
| S2                   | 1732.65 | 1780.72 | 567.44 | 8          | 303           | Compact                | Calibri           | Paper                 | Azure               | 4                | 12                | 8                  |
| S3                   | 115.66  | 100.76  | 64.68  | 9          | 302           | Compact                | Helvetica         | Digital               | Amber               | 1                | 27                | 4                  |

---

**Preserved Identifiers and Source Information:**
- **Facilities (Warehouses):** S1, S2, S3
- **Customers:** C1, C2, C3
- **Fixed Costs:** S1 (102.33), S2 (94.92), S3 (91.83)
- **Demands:** C1 (1083), C2 (776), C3 (16214)
- **Transportation Cost Matrix:** as above, with explicit facility and customer IDs, and source row positions.

**No capacity data is present; capacity is unresolved.**

**No extra axes or products are inferred.**  
**All data is preserved as per the original query.**