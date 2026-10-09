Below are all rows and columns from both service_centers_fixed_costs.csv and expanded_customer_service_costs.csv, preserving all facility and customer identifiers and values as requested.

---

**service_centers_fixed_costs.csv**

| Service Center | Fixed Opening Cost | fixed_opening_cost_previous_period | two_periods_ago_Fixed_Opening_Cost |
|:--------------:|:-----------------:|:----------------------------------:|:-----------------------------------:|
| SC1            | 385.1             | 383.75215                          | 427.92312                           |
| SC2            | 546.3             | 489.81258                          | 634.63671                           |
| SC3            | 485.2             | 428.14048                          | 549.68308                           |
| SC4            | 448.1             | 359.15215                          | 446.62127                           |
| SC5            | 324.1             | 276.87863                          | 309.35345                           |
| SC6            | 323.9             | 357.35887                          | 303.8182                            |
| SC7            | 296.5             | 341.95345                          | 316.3655                            |
| SC8            | 522.7             | 497.29678                          | 556.83231                           |
| SC9            | 448.7             | 373.99145                          | 482.12815                           |
| SC10           | 478.7             | 458.97756                          | 516.37369                           |

---

**expanded_customer_service_costs.csv**

| Customer | SC1   | SC2   | SC3   | SC4   | SC5   | SC6   | SC7   | SC8   | SC9   | SC10  |
|:--------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| C1       | 15.1  | 21.2  | 14.9  | 18.8  | 22.9  | 16.8  | 16.5  | 9.4   | 16.1  | 17.3  |
| C2       | 13.4  | 16.3  | 20.2  | 19.6  | 20.9  | 22.1  | 16.9  | 9.4   | 13.8  | 11.7  |
| C3       | 15.2  | 18.8  | 14.7  | 21.7  | 18.1  | 18.6  | 12.3  | 11.2  | 11.9  | 20.4  |
| C4       | 16.8  | 19.1  | 18.3  | 18.8  | 23.1  | 15.7  | 13.1  | 8.6   | 15.6  | 22.2  |
| C5       | 13.4  | 18.6  | 20.8  | 19.8  | 22.1  | 18.1  | 16.7  | 12.1  | 11.4  | 18.2  |
| C6       | 12.5  | 22.5  | 15.5  | 14.9  | 21.6  | 21.3  | 16.1  | 10.7  | 11.9  | 14.6  |
| C7       | 12.1  | 17.1  | 19.8  | 18.6  | 22.1  | 20.7  | 20.5  | 12.2  | 15.4  | 18.7  |
| C8       | 12.3  | 15.7  | 17.9  | 21.3  | 22.7  | 15.3  | 16.6  | 11.4  | 14.1  | 20.1  |
| C9       | 16.3  | 21.3  | 17.6  | 20.8  | 21.8  | 17.2  | 15.5  | 12.6  | 19.9  | 19.1  |
| C10      | 12.1  | 18.7  | 14.4  | 20.1  | 22.7  | 14.1  | 18.1  | 11.4  | 18.1  | 17.4  |
| C11      | 16.7  | 18.7  | 15.7  | 19.9  | 24.2  | 18.7  | 14.2  | 13.1  | 14.7  | 16.1  |
| C12      | 11.3  | 23.8  | 15.5  | 17.3  | 23.2  | 17.7  | 16.8  | 14.5  | 15.8  | 17.8  |
| C13      | 15.1  | 20.5  | 15.1  | 18.4  | 20.6  | 17.9  | 14.5  | 8.5   | 14.9  | 13.9  |
| C14      | 8.3   | 20.7  | 14.7  | 20.4  | 20.6  | 14.8  | 14.2  | 11.5  | 14.1  | 15.1  |
| C15      | 12.1  | 16.3  | 16.4  | 15.1  | 21.3  | 19.1  | 19.5  | 16.7  | 11.1  | 18.7  |

---

**Notes:**
- Each service centre (SC1–SC10) has a fixed opening cost as shown above.
- Each customer (C1–C15) has a service cost for being served from each centre (SC1–SC10) as shown in the matrix above.
- No capacity column is present, but the requirement is that each centre may serve at most 4 customers.
- All identifiers and values are preserved as in the original files. No data has been omitted, transposed, or inferred.