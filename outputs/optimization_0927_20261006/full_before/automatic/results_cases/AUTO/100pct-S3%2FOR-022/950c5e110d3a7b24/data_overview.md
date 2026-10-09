Below is all the data required to formulate the described facility location and transportation model, preserving the source order, exact identifiers, and values. Each entry retains its original facility/customer IDs, cost, and demand values, with explicit matrix axes and no inferred or omitted data.

---

### Demand Data (from 'demand.csv')
**Source row 1:**  
- customer_id: C1  
  - demand_units: 143  
  - demand_previous_period: 149  
  - two_periods_ago_demand_units: 133  

**Source row 2:**  
- customer_id: C2  
  - demand_units: 6  
  - demand_previous_period: 7  
  - two_periods_ago_demand_units: 5  

**Source row 3:**  
- customer_id: C3  
  - demand_units: 10  
  - demand_previous_period: 9  
  - two_periods_ago_demand_units: 11  

**Source row 4:**  
- customer_id: C4  
  - demand_units: 25  
  - demand_previous_period: 26  
  - two_periods_ago_demand_units: 26  

**Source row 5:**  
- customer_id: C5  
  - demand_units: 3  
  - demand_previous_period: 4  
  - two_periods_ago_demand_units: 4  

---

### Fixed Opening Cost Data (from 'fixed_cost.csv')
**Source row 1:**  
- facility_id: S1  
  - fixed_opening_cost: 97.65  
  - fixed_opening_cost_previous_period: 89.720820  
  - two_periods_ago_fixed_opening_cost: 109.260585  

**Source row 2:**  
- facility_id: S2  
  - fixed_opening_cost: 99.76  
  - fixed_opening_cost_previous_period: 115.033256  
  - two_periods_ago_fixed_opening_cost: 83.040224  

**Source row 3:**  
- facility_id: S3  
  - fixed_opening_cost: 100.76  
  - fixed_opening_cost_previous_period: 99.117612  
  - two_periods_ago_fixed_opening_cost: 87.25816  

**Source row 4:**  
- facility_id: S4  
  - fixed_opening_cost: 105.32  
  - fixed_opening_cost_previous_period: 86.867936  
  - two_periods_ago_fixed_opening_cost: 126.015380  

**Source row 5:**  
- facility_id: S5  
  - fixed_opening_cost: 98.88  
  - fixed_opening_cost_previous_period: 95.340096  
  - two_periods_ago_fixed_opening_cost: 90.346656  

---

### Transportation Cost Matrix (from 'transportation_costs.csv')
**Source row 1 (facility_id: S1):**  
- transportation_cost_to_C1: 150.74  
- transportation_cost_to_C2: 0.02  
- transportation_cost_to_C3: 49.13  
- transportation_cost_to_C4: 2080.15  
- transportation_cost_to_C5: 426.4  
- previous_period_transportation_cost_to_C1: 179.290156  
- previous_period_transportation_cost_to_C2: 0.017934  
- previous_period_transportation_cost_to_C3: 49.444432  
- previous_period_transportation_cost_to_C4: 2240.945595  
- two_periods_ago_service_status: Suspended  
- previous_period_service_status: Suspended  

**Source row 2 (facility_id: S2):**  
- transportation_cost_to_C1: 233.05  
- transportation_cost_to_C2: 97.73  
- transportation_cost_to_C3: 49.84  
- transportation_cost_to_C4: 1982.39  
- transportation_cost_to_C5: 23.96  
- previous_period_transportation_cost_to_C1: 233.609320  
- previous_period_transportation_cost_to_C2: 104.27791  
- previous_period_transportation_cost_to_C3: 56.46872  
- previous_period_transportation_cost_to_C4: 2042.456417  
- two_periods_ago_service_status: Seasonal  
- previous_period_service_status: Regular  

**Source row 3 (facility_id: S3):**  
- transportation_cost_to_C1: 55.68  
- transportation_cost_to_C2: 935.61  
- transportation_cost_to_C3: 4.03  
- transportation_cost_to_C4: 73.09  
- transportation_cost_to_C5: 525.32  
- previous_period_transportation_cost_to_C1: 51.910464  
- previous_period_transportation_cost_to_C2: 993.992064  
- previous_period_transportation_cost_to_C3: 4.53778  
- previous_period_transportation_cost_to_C4: 63.771025  
- two_periods_ago_service_status: Seasonal  
- previous_period_service_status: Trial  

**Source row 4 (facility_id: S4):**  
- transportation_cost_to_C1: 1483.82  
- transportation_cost_to_C2: 1801.08  
- transportation_cost_to_C3: 112.16  
- transportation_cost_to_C4: 816.05  
- transportation_cost_to_C5: 107.01  
- previous_period_transportation_cost_to_C1: 1536.050464  
- previous_period_transportation_cost_to_C2: 1495.436724  
- previous_period_transportation_cost_to_C3: 128.860624  
- previous_period_transportation_cost_to_C4: 659.613215  
- two_periods_ago_service_status: Suspended  
- previous_period_service_status: Trial  

**Source row 5 (facility_id: S5):**  
- transportation_cost_to_C1: 1119.47  
- transportation_cost_to_C2: 884.31  
- transportation_cost_to_C3: 0.08  
- transportation_cost_to_C4: 1544.95  
- transportation_cost_to_C5: 543.67  
- previous_period_transportation_cost_to_C1: 1191.787762  
- previous_period_transportation_cost_to_C2: 840.448224  
- previous_period_transportation_cost_to_C3: 0.093488  
- previous_period_transportation_cost_to_C4: 1416.564655  
- two_periods_ago_service_status: Seasonal  
- previous_period_service_status: Suspended  

---

**Note:**  
- No explicit capacity data is present for any facility; capacity is unresolved evidence, not zero.
- All identifiers and values are preserved as in the source.
- The cost matrix is facility (row) × customer (column), with explicit IDs.
- No product/commodity axis is present or inferred.

This data is sufficient to formulate the two-dimensional shipment decision model as described.