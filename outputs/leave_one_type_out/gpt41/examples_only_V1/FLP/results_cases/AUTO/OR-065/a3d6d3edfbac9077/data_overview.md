Here are all transportation costs per unit from each warehouse to each customer from "transportation_costs.csv", preserving the original matrix orientation and explicit facility/customer IDs:

**Source: transportation_costs.csv**

| Warehouse (Facility ID) | Customer (Customer ID) | Transportation Cost per Unit |
|------------------------|------------------------|-----------------------------|
| S1                     | C1                     | 1506.22                     |
| S1                     | C2                     | 70.90                       |
| S1                     | C3                     | 8.44                        |
| S2                     | C1                     | 1732.65                     |
| S2                     | C2                     | 1780.72                     |
| S2                     | C3                     | 567.44                      |
| S3                     | C1                     | 115.66                      |
| S3                     | C2                     | 100.76                      |
| S3                     | C3                     | 64.68                       |

- Facility IDs: S1, S2, S3
- Customer IDs: C1, C2, C3
- Matrix shape: 3 (facilities) × 3 (customers)
- Each entry is the cost per unit to ship from the given warehouse to the given customer.