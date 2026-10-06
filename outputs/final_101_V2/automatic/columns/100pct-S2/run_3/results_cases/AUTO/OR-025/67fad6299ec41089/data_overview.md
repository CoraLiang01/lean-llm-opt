Here is the complete retrieval of all relevant data from the provided context, preserving all identifiers, values, and source-row positions, as requested:

---

### fixed_cost.csv

| Facility ID | facility_staff_count | FixedCost | facility_reception_desk_count_2025_q4 | Source-row position |
|-------------|---------------------|-----------|---------------------------------------|--------------------|
| S1          | 65                  | 105.97    | 1                                     | 1                  |
| S2          | 20                  | 85.31     | 4                                     | 2                  |

---

### transportation_costs.csv

| Facility ID | operations_region | carrier_coordination_meeting_count_2025_q4 | annual_inspection_count | To Customer | Transportation Cost | Source-row position |
|-------------|------------------|--------------------------------------------|------------------------|-------------|--------------------|--------------------|
| S1          | West             | 2                                          | 1                      | C1          | 2358.39            | 1                  |
| S1          | West             | 2                                          | 1                      | C2          | 1492.08            | 1                  |
| S2          | North            | 3                                          | 4                      | C1          | 0.07               | 2                  |
| S2          | North            | 3                                          | 4                      | C2          | 52.32              | 2                  |

---

### demand.csv

| Customer ID | customer_support_ticket_count | Demand | customer_newsletter_open_count_2025_q4 | Source-row position |
|-------------|------------------------------|--------|----------------------------------------|--------------------|
| C1          | 3                            | 144    | 5                                      | 1                  |
| C2          | 1                            | 216    | 1                                      | 2                  |

---

**Notes:**
- Facility IDs: S1, S2
- Customer IDs: C1, C2
- FixedCost and facility attributes are matched by Facility ID.
- Transportation costs are provided for each (Facility, Customer) pair, with explicit matrix orientation.
- Demand is specified per customer.
- No explicit capacity is stated for facilities; this is unresolved evidence, not zero.
- All data is preserved as in the original context, with no transposition, truncation, or inference beyond the provided information.