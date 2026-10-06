**Retrieved Data**

---

### demand.csv

| Customer | Four Periods Ago Demand | Three Periods Ago Demand | Two Periods Ago Demand | Demand Previous Period | Demand (Current) |
|----------|------------------------|-------------------------|-----------------------|-----------------------|------------------|
| C1       | 954                    | 984                     | 900                   | 1006                  | 1083             |
| C2       | 811                    | 916                     | 622                   | 842                   | 776              |
| C3       | 13404                  | 17373                   | 13681                 | 16770                 | 16214            |

---

### fixed_cost.csv

| Facility (Warehouse) | Four Periods Ago Fixed Costs | Three Periods Ago Fixed Costs | Two Periods Ago Fixed Costs | Fixed Opening Cost Previous Period | Fixed Costs (Current) |
|----------------------|-----------------------------|------------------------------|-----------------------------|------------------------------------|-----------------------|
| S1                   | 114.435639                  | 107.95815                    | 122.120622                  | 86.734908                         | 102.33                |
| S2                   | 107.525376                  | 103.225500                   | 90.107556                   | 78.02424                          | 94.92                 |
| S3                   | 91.168824                   | 87.615003                    | 97.24797                    | 105.51267                         | 91.83                 |

---

### transportation_costs.csv

| Facility (Warehouse) | C1 (Current) | C2 (Current) | C3 (Current) | Previous Period C1 | Previous Period C2 | Previous Period C3 | Two Periods Ago C1 | Three Periods Ago Service Status | Four Periods Ago Service Status | Two Periods Ago Service Status | Previous Period Service Status |
|----------------------|--------------|--------------|--------------|--------------------|--------------------|--------------------|--------------------|-------------------------------|-------------------------------|------------------------------|------------------------------|
| S1                   | 1506.22      | 70.9         | 8.44         | 1364.785942        | 70.99217           | 7.958076           | 1233.744802        | Regular                       | Seasonal                      | Seasonal                     | Suspended                    |
| S2                   | 1732.65      | 1780.72      | 567.44       | 1958.414295        | 1932.793488        | 494.013264         | 1847.871225        | Seasonal                      | Regular                       | Trial                        | Seasonal                     |
| S3                   | 115.66       | 100.76       | 64.68        | 131.574816         | 91.258332          | 66.575124          | 122.981278         | Seasonal                      | Regular                       | Trial                        | Regular                      |

---

**Preserved Identifiers and Values:**

- **Facilities (Warehouses):** S1, S2, S3
- **Customers (Musicians/Bands):** C1, C2, C3
- **Fixed Costs:** For each facility, all periods retained, with current period as primary.
- **Demand:** For each customer, all periods retained, with current period as primary.
- **Transportation Cost Matrix:** For each facility-customer pair, all periods retained, with current period as primary. Matrix orientation is Facility (row) × Customer (column).

**No capacity data is present.**

**No data has been transposed, truncated, padded, zero-filled, or inferred beyond the provided context.**