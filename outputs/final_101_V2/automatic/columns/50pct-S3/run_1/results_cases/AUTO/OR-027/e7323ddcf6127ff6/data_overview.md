Retrieved Data

**From service_centers_fixed_costs.csv**  
(Fixed opening cost for each centre SC1–SC10)

| Service Center | Fixed Opening Cost | Source Row |
|---------------|-------------------|------------|
| SC1           | 385.1             | 1          |
| SC2           | 546.3             | 2          |
| SC3           | 485.2             | 3          |
| SC4           | 448.1             | 4          |
| SC5           | 324.1             | 5          |
| SC6           | 323.9             | 6          |
| SC7           | 296.5             | 7          |
| SC8           | 522.7             | 8          |
| SC9           | 448.7             | 9          |
| SC10          | 478.7             | 10         |

**From expanded_customer_service_costs.csv**  
(Per-customer service cost from each centre; matrix shape: 15 customers × 10 centres; row axis = Customer ID, column axis = Service Center ID)

| Customer | SC1   | SC2   | SC3   | SC4   | SC5   | SC6   | SC7   | SC8   | SC9   | SC10  | Source Row |
|----------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|------------|
| C1       | 15.1  | 21.2  | 14.9  | 18.8  | 22.9  | 16.8  | 16.5  | 9.4   | 16.1  | 17.3   | 1          |
| C2       | 13.4  | 16.3  | 20.2  | 19.6  | 20.9  | 22.1  | 16.9  | 9.4   | 13.8  | 11.7   | 2          |
| C3       | 15.2  | 18.8  | 14.7  | 21.7  | 18.1  | 18.6  | 12.3  | 11.2  | 11.9  | 20.4   | 3          |
| C4       | 16.8  | 19.1  | 18.3  | 18.8  | 23.1  | 15.7  | 13.1  | 8.6   | 15.6  | 22.2   | 4          |
| C5       | 13.4  | 18.6  | 20.8  | 19.8  | 22.1  | 18.1  | 16.7  | 12.1  | 11.4  | 18.2   | 5          |
| C6       | 12.5  | 22.5  | 15.5  | 14.9  | 21.6  | 21.3  | 16.1  | 10.7  | 11.9  | 14.6   | 6          |
| C7       | 12.1  | 17.1  | 19.8  | 18.6  | 22.1  | 20.7  | 20.5  | 12.2  | 15.4  | 18.7   | 7          |
| C8       | 12.3  | 15.7  | 17.9  | 21.3  | 22.7  | 15.3  | 16.6  | 11.4  | 14.1  | 20.1   | 8          |
| C9       | 16.3  | 21.3  | 17.6  | 20.8  | 21.8  | 17.2  | 15.5  | 12.6  | 19.9  | 19.1   | 9          |
| C10      | 12.1  | 18.7  | 14.4  | 20.1  | 22.7  | 14.1  | 18.1  | 11.4  | 18.1  | 17.4   | 10         |
| C11      | 16.7  | 18.7  | 15.7  | 19.9  | 24.2  | 18.7  | 14.2  | 13.1  | 14.7  | 16.1   | 11         |
| C12      | 11.3  | 23.8  | 15.5  | 17.3  | 23.2  | 17.7  | 16.8  | 14.5  | 15.8  | 17.8   | 12         |
| C13      | 15.1  | 20.5  | 15.1  | 18.4  | 20.6  | 17.9  | 14.5  | 8.5   | 14.9  | 13.9   | 13         |
| C14      | 8.3   | 20.7  | 14.7  | 20.4  | 20.6  | 14.8  | 14.2  | 11.5  | 14.1  | 15.1   | 14         |
| C15      | 12.1  | 16.3  | 16.4  | 15.1  | 21.3  | 19.1  | 19.5  | 16.7  | 11.1  | 18.7   | 15         |

**Assignment Capacity Constraint:**  
Each opened centre may serve at most 4 customers.

**Summary of Preserved Data:**
- Facility IDs: SC1–SC10, with fixed opening costs as above.
- Customer IDs: C1–C15.
- Cost matrix: 15 (customers) × 10 (centres), as above.
- Capacity: Each centre (SC1–SC10) may serve at most 4 customers.
- All axes and identifiers are preserved as in the source. No transposition, truncation, or inference applied.