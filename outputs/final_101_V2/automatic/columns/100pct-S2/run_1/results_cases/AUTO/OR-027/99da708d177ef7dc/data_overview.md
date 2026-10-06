Retrieved Data

service_centers_fixed_costs.csv  
(Source: Service Center fixed cost table; each row is a facility, with its ID and fixed opening cost.)

| Facility ID | FixedCost | facility_staff_count | facility_reception_desk_count_2025_q4 | Source Row |
|-------------|-----------|---------------------|---------------------------------------|------------|
| SC1         | 385.1     | 35                  | 2                                     | SC1        |
| SC2         | 546.3     | 50                  | 3                                     | SC2        |
| SC3         | 485.2     | 35                  | 2                                     | SC3        |
| SC4         | 448.1     | 12                  | 3                                     | SC4        |
| SC5         | 324.1     | 35                  | 4                                     | SC5        |
| SC6         | 323.9     | 20                  | 4                                     | SC6        |
| SC7         | 296.5     | 65                  | 3                                     | SC7        |
| SC8         | 522.7     | 35                  | 3                                     | SC8        |
| SC9         | 448.7     | 8                   | 2                                     | SC9        |
| SC10        | 478.7     | 20                  | 1                                     | SC10       |

Capacity constraint: Each opened centre may serve at most 4 customers.

expanded_customer_service_costs.csv  
(Source: Customer-to-centre service cost matrix; each row is a customer, each column is a facility. All values are preserved as given.)

| Customer | SC1  | SC2  | SC3  | SC4  | SC5  | SC6  | SC7  | SC8  | SC9  | SC10 | Source Row |
|----------|------|------|------|------|------|------|------|------|------|-------|------------|
| C1       | 15.1 | 21.2 | 14.9 | 18.8 | 22.9 | 16.8 | 16.5 | 9.4  | 16.1 | 17.3 | C1         |
| C2       | 13.4 | 16.3 | 20.2 | 19.6 | 20.9 | 22.1 | 16.9 | 9.4  | 13.8 | 11.7 | C2         |
| C3       | 15.2 | 18.8 | 14.7 | 21.7 | 18.1 | 18.6 | 12.3 | 11.2 | 11.9 | 20.4 | C3         |
| C4       | 16.8 | 19.1 | 18.3 | 18.8 | 23.1 | 15.7 | 13.1 | 8.6  | 15.6 | 22.2 | C4         |
| C5       | 13.4 | 18.6 | 20.8 | 19.8 | 22.1 | 18.1 | 16.7 | 12.1 | 11.4 | 18.2 | C5         |
| C6       | 12.5 | 22.5 | 15.5 | 14.9 | 21.6 | 21.3 | 16.1 | 10.7 | 11.9 | 14.6 | C6         |
| C7       | 12.1 | 17.1 | 19.8 | 18.6 | 22.1 | 20.7 | 20.5 | 12.2 | 15.4 | 18.7 | C7         |
| C8       | 12.3 | 15.7 | 17.9 | 21.3 | 22.7 | 15.3 | 16.6 | 11.4 | 14.1 | 20.1 | C8         |
| C9       | 16.3 | 21.3 | 17.6 | 20.8 | 21.8 | 17.2 | 15.5 | 12.6 | 19.9 | 19.1 | C9         |
| C10      | 12.1 | 18.7 | 14.4 | 20.1 | 22.7 | 14.1 | 18.1 | 11.4 | 18.1 | 17.4 | C10        |
| C11      | 16.7 | 18.7 | 15.7 | 19.9 | 24.2 | 18.7 | 14.2 | 13.1 | 14.7 | 16.1 | C11        |
| C12      | 11.3 | 23.8 | 15.5 | 17.3 | 23.2 | 17.7 | 16.8 | 14.5 | 15.8 | 17.8 | C12        |
| C13      | 15.1 | 20.5 | 15.1 | 18.4 | 20.6 | 17.9 | 14.5 | 8.5  | 14.9 | 13.9 | C13        |
| C14      | 8.3  | 20.7 | 14.7 | 20.4 | 20.6 | 14.8 | 14.2 | 11.5 | 14.1 | 15.1 | C14        |
| C15      | 12.1 | 16.3 | 16.4 | 15.1 | 21.3 | 19.1 | 19.5 | 16.7 | 11.1 | 18.7 | C15        |

All facility IDs, customer IDs, fixed costs, and the full cost matrix are preserved as required. No data has been omitted, transposed, or inferred. Capacity per facility is 4 customers.