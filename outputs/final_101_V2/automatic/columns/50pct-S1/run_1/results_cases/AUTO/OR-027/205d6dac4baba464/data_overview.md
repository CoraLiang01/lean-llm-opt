Below is the complete retrieval of all relevant data from both service_centers_fixed_costs.csv and expanded_customer_service_costs.csv, preserving all facility (service centre) and customer IDs, fixed costs, and the full cost matrix. All identifiers and values are retained as in the source, with explicit axis orientation and no inferred or omitted data.

---

### Service Centre Fixed Opening Costs  
(Source: service_centers_fixed_costs.csv)

| Facility ID | Fixed Opening Cost | Source Row |
|-------------|-------------------|------------|
| SC1         | 385.1             | 1          |
| SC2         | 546.3             | 2          |
| SC3         | 485.2             | 3          |
| SC4         | 448.1             | 4          |
| SC5         | 324.1             | 5          |
| SC6         | 323.9             | 6          |
| SC7         | 296.5             | 7          |
| SC8         | 522.7             | 8          |
| SC9         | 448.7             | 9          |
| SC10        | 478.7             | 10         |

---

### Customer–Service Centre Assignment Cost Matrix  
(Source: expanded_customer_service_costs.csv)

- **Rows:** Customers (C1–C15)
- **Columns:** Service Centres (SC1–SC10)
- **Cell [i, j]:** Cost to serve customer Ci from centre SCj

| Customer | SC1   | SC2   | SC3   | SC4   | SC5   | SC6   | SC7   | SC8   | SC9   | SC10  | Source Row |
|----------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|------------|
| C1       | 15.1  | 21.2  | 14.9  | 18.8  | 22.9  | 16.8  | 16.5  | 9.4   | 16.1  | 17.3  | 1          |
| C2       | 13.4  | 16.3  | 20.2  | 19.6  | 20.9  | 22.1  | 16.9  | 9.4   | 13.8  | 11.7  | 2          |
| C3       | 15.2  | 18.8  | 14.7  | 21.7  | 18.1  | 18.6  | 12.3  | 11.2  | 11.9  | 20.4  | 3          |
| C4       | 16.8  | 19.1  | 18.3  | 18.8  | 23.1  | 15.7  | 13.1  | 8.6   | 15.6  | 22.2  | 4          |
| C5       | 13.4  | 18.6  | 20.8  | 19.8  | 22.1  | 18.1  | 16.7  | 12.1  | 11.4  | 18.2  | 5          |
| C6       | 12.5  | 22.5  | 15.5  | 14.9  | 21.6  | 21.3  | 16.1  | 10.7  | 11.9  | 14.6  | 6          |
| C7       | 12.1  | 17.1  | 19.8  | 18.6  | 22.1  | 20.7  | 20.5  | 12.2  | 15.4  | 18.7  | 7          |
| C8       | 12.3  | 15.7  | 17.9  | 21.3  | 22.7  | 15.3  | 16.6  | 11.4  | 14.1  | 20.1  | 8          |
| C9       | 16.3  | 21.3  | 17.6  | 20.8  | 21.8  | 17.2  | 15.5  | 12.6  | 19.9  | 19.1  | 9          |
| C10      | 12.1  | 18.7  | 14.4  | 20.1  | 22.7  | 14.1  | 18.1  | 11.4  | 18.1  | 17.4  | 10         |
| C11      | 16.7  | 18.7  | 15.7  | 19.9  | 24.2  | 18.7  | 14.2  | 13.1  | 14.7  | 16.1  | 11         |
| C12      | 11.3  | 23.8  | 15.5  | 17.3  | 23.2  | 17.7  | 16.8  | 14.5  | 15.8  | 17.8  | 12         |
| C13      | 15.1  | 20.5  | 15.1  | 18.4  | 20.6  | 17.9  | 14.5  | 8.5   | 14.9  | 13.9  | 13         |
| C14      | 8.3   | 20.7  | 14.7  | 20.4  | 20.6  | 14.8  | 14.2  | 11.5  | 14.1  | 15.1  | 14         |
| C15      | 12.1  | 16.3  | 16.4  | 15.1  | 21.3  | 19.1  | 19.5  | 16.7  | 11.1  | 18.7  | 15         |

---

#### Notes:
- **Facility IDs:** SC1–SC10
- **Customer IDs:** C1–C15
- **Fixed Opening Cost:** As above, per facility
- **Assignment Cost Matrix:** As above, [Customer, Service Centre] orientation
- **Capacity:** Each centre may serve at most 4 customers (as per requirements)
- **No capacity values are present in the data; the only constraint is the maximum of 4 customers per centre.**

All data is preserved as in the original files, with explicit axis and source row positions. No data has been omitted, inferred, or transformed.