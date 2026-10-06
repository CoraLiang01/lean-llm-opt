**Retrieved Data**

---

### 1. Demand Data (`demand.csv`)

| Customer ID | Demand |
|-------------|--------|
| C1          | 1083   |
| C2          | 776    |
| C3          | 16214  |

**Source row positions:**  
- C1: Row 1  
- C2: Row 2  
- C3: Row 3  

---

### 2. Fixed Cost Data (`fixed_cost.csv`)

| Facility ID | Fixed Cost | Source Row Position |
|-------------|------------|--------------------|
| S1          | 102.33     | Row 1              |
| S2          | 94.92      | Row 2              |
| S3          | 91.83      | Row 3              |

---

### 3. Transportation Cost Matrix (`transportation_costs.csv`)

| Facility ID | Customer ID | Cost per Unit | Source Orientation (Facility→Customer) | Source Row (Facility) | Source Column (Customer) |
|-------------|-------------|--------------|----------------------------------------|-----------------------|--------------------------|
| S1          | C1          | 1506.22      | S1→C1                                  | S1                    | C1                       |
| S1          | C2          | 70.9         | S1→C2                                  | S1                    | C2                       |
| S1          | C3          | 8.44         | S1→C3                                  | S1                    | C3                       |
| S2          | C1          | 1732.65      | S2→C1                                  | S2                    | C1                       |
| S2          | C2          | 1780.72      | S2→C2                                  | S2                    | C2                       |
| S2          | C3          | 567.44       | S2→C3                                  | S2                    | C3                       |
| S3          | C1          | 115.66       | S3→C1                                  | S3                    | C1                       |
| S3          | C2          | 100.76       | S3→C2                                  | S3                    | C2                       |
| S3          | C3          | 64.68        | S3→C3                                  | S3                    | C3                       |

---

**Summary Table (for clarity):**

|           | C1      | C2      | C3      | Fixed Cost |
|-----------|---------|---------|---------|------------|
| **S1**    | 1506.22 | 70.9    | 8.44    | 102.33     |
| **S2**    | 1732.65 | 1780.72 | 567.44  | 94.92      |
| **S3**    | 115.66  | 100.76  | 64.68   | 91.83      |

- **Facility IDs:** S1, S2, S3
- **Customer IDs:** C1, C2, C3
- **Fixed Costs:** Explicitly matched to each facility
- **Demand:** Explicitly matched to each customer
- **Transportation Cost Matrix:** Facility (row) to Customer (column), as in the original data

---

**No capacity data is present; capacity is unresolved.**  
**No extra axes or products are present.**  
**All identifiers and source positions are preserved.**