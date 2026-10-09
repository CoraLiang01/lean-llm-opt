**Retrieved Data**

---

### demand.csv

| Customer | Demand | Previous Period Demand | Two Periods Ago Demand | Three Periods Ago Demand | Four Periods Ago Demand |
|----------|--------|-----------------------|------------------------|-------------------------|------------------------|
| C1       | 1083   | 1006                  | 900                    | 984                     | 954                    |
| C2       | 776    | 842                   | 622                    | 916                     | 811                    |
| C3       | 16214  | 16770                 | 13681                  | 17373                   | 13404                  |

---

### fixed_cost.csv

| Facility (Unnamed: 0) | Fixed Cost | Fixed Opening Cost Previous Period | Two Periods Ago Fixed Costs | Three Periods Ago Fixed Costs | Four Periods Ago Fixed Costs |
|-----------------------|------------|------------------------------------|----------------------------|------------------------------|-----------------------------|
| S1                    | 102.33     | 86.734908                          | 122.120622                 | 107.95815                    | 114.435639                  |
| S2                    | 94.92      | 78.02424                           | 90.107556                  | 103.225500                   | 107.525376                  |
| S3                    | 91.83      | 105.51267                          | 97.24797                   | 87.615003                    | 91.168824                   |

---

### transportation_costs.csv

| Facility (Unnamed: 0) | C1        | C2        | C3      | Previous Period C1 | Previous Period C2 | Previous Period C3 | Two Periods Ago C1 | Previous Period Service Status | Two Periods Ago Service Status | Three Periods Ago Service Status | Four Periods Ago Service Status |
|-----------------------|-----------|-----------|---------|--------------------|--------------------|--------------------|--------------------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|
| S1                    | 1506.22   | 70.9      | 8.44    | 1364.785942        | 70.99217           | 7.958076           | 1233.744802        | Suspended                     | Seasonal                      | Regular                       | Seasonal                      |
| S2                    | 1732.65   | 1780.72   | 567.44  | 1958.414295        | 1932.793488        | 494.013264         | 1847.871225        | Seasonal                      | Trial                         | Seasonal                      | Regular                       |
| S3                    | 115.66    | 100.76    | 64.68   | 131.574816         | 91.258332          | 66.575124          | 122.981278         | Regular                       | Trial                         | Seasonal                      | Regular                       |

---

**Preserved Identifiers and Source Row/Column Positions:**

- **Facilities (Warehouses):** S1, S2, S3
- **Customers (Musicians/Bands):** C1, C2, C3
- **Fixed Costs:** Matched to each facility (S1, S2, S3)
- **Demand:** Matched to each customer (C1, C2, C3)
- **Transportation Cost Matrix:** Rows = Facilities (S1, S2, S3), Columns = Customers (C1, C2, C3), with additional historical and status columns preserved as in source.

**No capacity data is present; capacity is unresolved.**

**No data has been transposed, truncated, or inferred beyond the original files.**