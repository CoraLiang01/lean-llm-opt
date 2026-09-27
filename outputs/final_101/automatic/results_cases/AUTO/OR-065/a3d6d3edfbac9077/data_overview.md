Here is the complete retrieval of all data from 'fixed_cost.csv', 'transportation_costs.csv', and 'demand.csv', with all identifiers and values preserved:

---

### demand.csv

| customer | demand |
|----------|--------|
| C1       | 1083   |  *(source-row: 1)*  
| C2       | 776    |  *(source-row: 2)*  
| C3       | 16214  |  *(source-row: 3)*  

---

### fixed_cost.csv

| Facility ID (Unnamed: 0) | FixedCost |
|--------------------------|-----------|
| S1                       | 102.33    |  *(source-row: 4)*  
| S2                       | 94.92     |  *(source-row: 5)*  
| S3                       | 91.83     |  *(source-row: 6)*  

---

### transportation_costs.csv

| Facility ID (Unnamed: 0) | C1      | C2      | C3      |
|--------------------------|---------|---------|---------|
| S1                       | 1506.22 | 70.90   | 8.44    |  *(source-row: 7)*  
| S2                       | 1732.65 | 1780.72 | 567.44  |  *(source-row: 8)*  
| S3                       | 115.66  | 100.76  | 64.68   |  *(source-row: 9)*  

---

**All facility IDs, customer IDs, fixed costs, and demand values are preserved. The cost-matrix axis is explicitly matched: rows are facilities (S1, S2, S3), columns are customers (C1, C2, C3), with source orientation and shape retained. No capacity data is present.**