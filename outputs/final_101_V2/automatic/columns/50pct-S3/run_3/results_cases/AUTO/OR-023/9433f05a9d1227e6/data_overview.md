Below is the complete retrieval of all data from the provided context, preserving all identifiers and values from the three files: fixed_cost.csv, transportation_costs.csv, and demand.csv. Each facility (supplier) and customer (store) is retained with its associated fixed cost, demand, and transportation cost matrix, with all source-row positions and axis labels preserved.

---

### fixed_cost.csv

| Facility ID   | Fixed Opening Cost Previous Period | Fixed Costs |
|---------------|-----------------------------------|-------------|
| MOUNT AYR     | 101.457290                        | 96.58       |
| WAUKEE        | 112.034866                        | 94.06       |
| WAVERLY       | 86.06544                          | 94.37       |
| PELLA         | 89.526976                         | 82.88       |
| DES MOINES    | 110.894288000                     | 94.96       |

---

### demand.csv

| Customer ID  | Demand Previous Period | Demand |
|--------------|-----------------------|--------|
| Customer_1   | 2025                  | 2397   |
| Customer_2   | 1729                  | 1889   |
| Customer_3   | 2280                  | 2518   |
| Customer_4   | 3008                  | 3218   |
| Customer_5   | 1814                  | 1813   |

---

### transportation_costs.csv

#### Row: MOUNT AYR

| Facility (Row) | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT | previous_period_FORT_MADISON | previous_period_CLARINDA | previous_period_service_status |
|----------------|----------|--------------|------------|--------|----------|------------------------------|--------------------------|-------------------------------|
| MOUNT AYR      | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  | 19.0095                      | 832.712916               | Seasonal                      |

#### Row: WAUKEE

| Facility (Row) | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT | previous_period_FORT_MADISON | previous_period_CLARINDA | previous_period_service_status |
|----------------|----------|--------------|------------|--------|----------|------------------------------|--------------------------|-------------------------------|
| WAUKEE         | 15.13    | 1.5          | 1.43       | 27.88  | 90.69    | 1.75905                      | 13.736527                | Trial                         |

#### Row: WAVERLY

| Facility (Row) | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT | previous_period_FORT_MADISON | previous_period_CLARINDA | previous_period_service_status |
|----------------|----------|--------------|------------|--------|----------|------------------------------|--------------------------|-------------------------------|
| WAVERLY        | 2.34     | 349.34       | 246.6      | 41.3   | 78.73    | 342.038794                   | 1.975662                 | Regular                       |

#### Row: PELLA

| Facility (Row) | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT | previous_period_FORT_MADISON | previous_period_CLARINDA | previous_period_service_status |
|----------------|----------|--------------|------------|--------|----------|------------------------------|--------------------------|-------------------------------|
| PELLA          | 1181.6   | 1458.53      | 1646.36    | 1924.55| 38.93    | 1520.663378                  | 960.05                   | Regular                       |

#### Row: DES MOINES

| Facility (Row) | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT | previous_period_FORT_MADISON | previous_period_CLARINDA | previous_period_service_status |
|----------------|----------|--------------|------------|--------|----------|------------------------------|--------------------------|-------------------------------|
| DES MOINES     | 1030.8   | 43.48        | 932.43     | 55.39  | 103.84   | 43.727836                    | 1093.98804               | Regular                       |

---

**Note:**  
- Facility IDs: MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES  
- Customer IDs (from demand.csv): Customer_1, Customer_2, Customer_3, Customer_4, Customer_5  
- Transportation cost matrix columns: CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT (these are likely store locations, to be mapped to Customer IDs as per the original data structure).
- All source-row positions and axis labels are preserved as in the original files.

If you need the data in a specific format (e.g., as a matrix or for a particular modeling tool), please specify.