**PotentialWarehouses_Costs.csv**

| Row | Warehouse (i) | Opening Cost (fi) | Capacity (units) | warehouse_roof_material | warehouse_inspection_count_2025_q4 |
|-----|---------------|-------------------|------------------|------------------------|-------------------------------------|
| 1   | 1             | 3000              | 180              | Concrete               | 6                                   |
| 2   | 2             | 3200              | 160              | Concrete               | 4                                   |
| 3   | 3             | 3100              | 200              | Steel                  | 4                                   |
| 4   | 4             | 2800              | 150              | Composite              | 2                                   |
| 5   | 5             | 3500              | 170              | Composite              | 6                                   |
| 6   | 6             | 2700              | 190              | Composite              | 3                                   |
| 7   | 7             | 2900              | 160              | Steel                  | 1                                   |
| 8   | 8             | 3050              | 175              | Steel                  | 2                                   |
| 9   | 9             | 3100              | 170              | Steel                  | 1                                   |
| 10  | 10            | 2200              | 180              | Concrete               | 1                                   |
| 11  | 11            | 2890              | 190              | Steel                  | 3                                   |

---

**Stores_Demands.csv**

| Row | Store (j) | Demand (units, dj) | store_staff_training_hours_2025_q4 |
|-----|-----------|--------------------|------------------------------------|
| 1   | 1         | 30                 | 12                                 |
| 2   | 2         | 40                 | 48                                 |
| 3   | 3         | 20                 | 18                                 |
| 4   | 4         | 35                 | 48                                 |
| 5   | 5         | 20                 | 18                                 |
| 6   | 6         | 25                 | 36                                 |
| 7   | 7         | 45                 | 18                                 |
| 8   | 8         | 38                 | 48                                 |
| 9   | 9         | 32                 | 12                                 |
| 10  | 10        | 41                 | 18                                 |
| 11  | 11        | 44                 | 36                                 |

---

**TransportationCost.csv**

*Each row is for a warehouse (W1 = 1, W2 = 2, ..., W11 = 11), each column is a store (1–11).*

| Row | Warehouse (i) | Store 1 | Store 2 | Store 3 | Store 4 | Store 5 | Store 6 | Store 7 | Store 8 | Store 9 | Store 10 | Store 11 | cost_record_route_survey_count_2025_q4 | cost_record_dispatch_coordination_meeting_count_2025_q4 | cost_record_logistics_training_hours_2025_q4 | cost_record_carrier_briefing_count_2025_q4 | cost_record_carrier_contact_channel | cost_record_tariff_review_meeting_count_2025_q4 |
|-----|---------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|-----------|-----------|----------------------------------------|--------------------------------------------------------|----------------------------------------------|---------------------------------------------|-------------------------------|-----------------------------------------------|
| 1   | 1             | 12      | 11      | 14      | 15      | 17      | 13      | 12      | 16      | 16      | 14        | 15        | 3                                      | 6                                                      | 12                                           | 3                                           | Email                         | 8                                             |
| 2   | 2             | 17      | 19      | 15      | 20      | 18      | 14      | 17      | 15      | 13      | 15        | 16        | 7                                      | 2                                                      | 12                                           | 2                                           | Telephone                     | 4                                             |
| 3   | 3             | 13      | 14      | 12      | 14      | 16      | 15      | 11      | 14      | 16      | 18        | 17        | 5                                      | 6                                                      | 30                                           | 1                                           | Telephone                     | 6                                             |
| 4   | 4             | 18      | 16      | 17      | 13      | 18      | 17      | 14      | 19      | 16      | 13        | 18        | 2                                      | 2                                                      | 24                                           | 4                                           | Telephone                     | 6                                             |
| 5   | 5             | 10      | 13      | 12      | 19      | 15      | 11      | 12      | 14      | 12      | 15        | 17        | 7                                      | 4                                                      | 18                                           | 4                                           | Email                         | 4                                             |
| 6   | 6             | 15      | 12      | 14      | 16      | 13      | 17      | 16      | 16      | 14      | 18        | 19        | 3                                      | 2                                                      | 12                                           | 1                                           | Portal                        | 1                                             |
| 7   | 7             | 14      | 13      | 15      | 17      | 12      | 13      | 14      | 15      | 12      | 16        | 14        | 2                                      | 2                                                      | 18                                           | 3                                           | Telephone                     | 1                                             |
| 8   | 8             | 19      | 16      | 18      | 20      | 17      | 19      | 16      | 18      | 15      | 15        | 18        | 7                                      | 4                                                      | 24                                           | 1                                           | Email                         | 4                                             |
| 9   | 9             | 17      | 18      | 12      | 14      | 16      | 15      | 14      | 17      | 21      | 15        | 18        | 1                                      | 4                                                      | 18                                           | 4                                           | Portal                        | 8                                             |
| 10  | 10            | 14      | 13      | 15      | 17      | 16      | 18      | 14      | 19      | 15      | 17        | 19        | 1                                      | 10                                                     | 12                                           | 2                                           | Portal                        | 8                                             |
| 11  | 11            | 15      | 13      | 16      | 17      | 11      | 13      | 14      | 15      | 19      | 21        | 13        | 2                                      | 10                                                     | 36                                           | 1                                           | Portal                        | 4                                             |

---

**Preserved Identifiers and Structure:**

- **Facility IDs:** 1–11 (Warehouse (i)), with Opening Cost and Capacity per warehouse.
- **Customer IDs:** 1–11 (Store (j)), with Demand per store.
- **Cost Matrix:** Transportation costs c_ij, with rows as warehouses (i=1..11), columns as stores (j=1..11), matching the original orientation.
- **No extra axes or inferred products.**
- **All data fields and row positions preserved as in the source.**