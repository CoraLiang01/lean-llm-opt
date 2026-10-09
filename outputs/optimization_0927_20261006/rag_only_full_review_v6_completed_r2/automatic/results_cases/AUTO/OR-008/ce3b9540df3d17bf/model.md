Sets:
- Let S = {Supplier1, Supplier2, Supplier3, Supplier4, Supplier5} (warehouses)
- Let C = {Customer1, Customer2, Customer3, Customer4, Customer5, Customer6} (stores)

Parameters:
- demand_c: daily demand for each store c ∈ C
  - demand_Customer1 = 70
  - demand_Customer2 = 80
  - demand_Customer3 = 60
  - demand_Customer4 = 90
  - demand_Customer5 = 85
  - demand_Customer6 = 95

- supply_capacity_s: supply capacity for each warehouse s ∈ S
  - supply_capacity_Supplier1 = 200
  - supply_capacity_Supplier2 = 250
  - supply_capacity_Supplier3 = 230
  - supply_capacity_Supplier4 = 220
  - supply_capacity_Supplier5 = 210

- transportation_cost_{s,c}: cost per unit from warehouse s to store c

|              | Customer1 | Customer2 | Customer3 | Customer4 | Customer5 | Customer6 |
|--------------|-----------|-----------|-----------|-----------|-----------|-----------|
| Supplier1    |     2     |     3     |     1     |     2     |     3     |     2     |
| Supplier2    |     1     |     2     |     3     |     2     |     3     |     2     |
| Supplier3    |     3     |     1     |     2     |     3     |     2     |     3     |
| Supplier4    |     2     |     3     |     2     |     1     |     3     |     4     |
| Supplier5    |     3     |     2     |     3     |     3     |     2     |     3     |

Decision Variables:
- x_{s,c} ≥ 0: amount of fresh produce shipped from warehouse s ∈ S to store c ∈ C

Objective:
Minimize total transportation cost:
minimize
  2 x_{Supplier1,Customer1} + 3 x_{Supplier1,Customer2} + 1 x_{Supplier1,Customer3} + 2 x_{Supplier1,Customer4} + 3 x_{Supplier1,Customer5} + 2 x_{Supplier1,Customer6}
+ 1 x_{Supplier2,Customer1} + 2 x_{Supplier2,Customer2} + 3 x_{Supplier2,Customer3} + 2 x_{Supplier2,Customer4} + 3 x_{Supplier2,Customer5} + 2 x_{Supplier2,Customer6}
+ 3 x_{Supplier3,Customer1} + 1 x_{Supplier3,Customer2} + 2 x_{Supplier3,Customer3} + 3 x_{Supplier3,Customer4} + 2 x_{Supplier3,Customer5} + 3 x_{Supplier3,Customer6}
+ 2 x_{Supplier4,Customer1} + 3 x_{Supplier4,Customer2} + 2 x_{Supplier4,Customer3} + 1 x_{Supplier4,Customer4} + 3 x_{Supplier4,Customer5} + 4 x_{Supplier4,Customer6}
+ 3 x_{Supplier5,Customer1} + 2 x_{Supplier5,Customer2} + 3 x_{Supplier5,Customer3} + 3 x_{Supplier5,Customer4} + 2 x_{Supplier5,Customer5} + 3 x_{Supplier5,Customer6}

Subject to:

1. Demand satisfaction for each store:
   For each c ∈ C,
   x_{Supplier1,c} + x_{Supplier2,c} + x_{Supplier3,c} + x_{Supplier4,c} + x_{Supplier5,c} = demand_c

   - x_{Supplier1,Customer1} + x_{Supplier2,Customer1} + x_{Supplier3,Customer1} + x_{Supplier4,Customer1} + x_{Supplier5,Customer1} = 70
   - x_{Supplier1,Customer2} + x_{Supplier2,Customer2} + x_{Supplier3,Customer2} + x_{Supplier4,Customer2} + x_{Supplier5,Customer2} = 80
   - x_{Supplier1,Customer3} + x_{Supplier2,Customer3} + x_{Supplier3,Customer3} + x_{Supplier4,Customer3} + x_{Supplier5,Customer3} = 60
   - x_{Supplier1,Customer4} + x_{Supplier2,Customer4} + x_{Supplier3,Customer4} + x_{Supplier4,Customer4} + x_{Supplier5,Customer4} = 90
   - x_{Supplier1,Customer5} + x_{Supplier2,Customer5} + x_{Supplier3,Customer5} + x_{Supplier4,Customer5} + x_{Supplier5,Customer5} = 85
   - x_{Supplier1,Customer6} + x_{Supplier2,Customer6} + x_{Supplier3,Customer6} + x_{Supplier4,Customer6} + x_{Supplier5,Customer6} = 95

2. Supply capacity for each warehouse:
   For each s ∈ S,
   x_{s,Customer1} + x_{s,Customer2} + x_{s,Customer3} + x_{s,Customer4} + x_{s,Customer5} + x_{s,Customer6} ≤ supply_capacity_s

   - x_{Supplier1,Customer1} + x_{Supplier1,Customer2} + x_{Supplier1,Customer3} + x_{Supplier1,Customer4} + x_{Supplier1,Customer5} + x_{Supplier1,Customer6} ≤ 200
   - x_{Supplier2,Customer1} + x_{Supplier2,Customer2} + x_{Supplier2,Customer3} + x_{Supplier2,Customer4} + x_{Supplier2,Customer5} + x_{Supplier2,Customer6} ≤ 250
   - x_{Supplier3,Customer1} + x_{Supplier3,Customer2} + x_{Supplier3,Customer3} + x_{Supplier3,Customer4} + x_{Supplier3,Customer5} + x_{Supplier3,Customer6} ≤ 230
   - x_{Supplier4,Customer1} + x_{Supplier4,Customer2} + x_{Supplier4,Customer3} + x_{Supplier4,Customer4} + x_{Supplier4,Customer5} + x_{Supplier4,Customer6} ≤ 220
   - x_{Supplier5,Customer1} + x_{Supplier5,Customer2} + x_{Supplier5,Customer3} + x_{Supplier5,Customer4} + x_{Supplier5,Customer5} + x_{Supplier5,Customer6} ≤ 210

3. Non-negativity:
   For all s ∈ S, c ∈ C:
   x_{s,c} ≥ 0

All data and coefficients are preserved in source order as required.