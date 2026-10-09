Let:
- S = {Supplier1, Supplier2, Supplier3, Supplier4, Supplier5} (warehouses)
- C = {Customer1, Customer2, Customer3, Customer4, Customer5, Customer6} (stores)
- x_{s,c} = amount of fresh produce shipped from supplier s to customer c (decision variables, continuous, x_{s,c} ≥ 0)

Parameters (from CSVs, source order preserved):

Customer demands (customer_demand.csv):
- demand_{Customer1} = 70
- demand_{Customer2} = 80
- demand_{Customer3} = 60
- demand_{Customer4} = 90
- demand_{Customer5} = 85
- demand_{Customer6} = 95

Supplier capacities (supply_capacity.csv):
- supply_capacity_{Supplier1} = 200
- supply_capacity_{Supplier2} = 250
- supply_capacity_{Supplier3} = 230
- supply_capacity_{Supplier4} = 220
- supply_capacity_{Supplier5} = 210

Transportation costs per unit (transportation_costs.csv):

|            | Customer1 | Customer2 | Customer3 | Customer4 | Customer5 | Customer6 |
|------------|-----------|-----------|-----------|-----------|-----------|-----------|
| Supplier1  |     2     |     3     |     1     |     2     |     3     |     2     |
| Supplier2  |     1     |     2     |     3     |     2     |     3     |     2     |
| Supplier3  |     3     |     1     |     2     |     3     |     2     |     3     |
| Supplier4  |     2     |     3     |     2     |     1     |     3     |     4     |
| Supplier5  |     3     |     2     |     3     |     3     |     2     |     3     |

Model:

Variables:
- x_{s,c} ≥ 0, ∀ s ∈ S, c ∈ C

Objective (minimize total transportation cost, preserving full expression and constant terms):

Minimize
Z = 2 x_{Supplier1,Customer1} + 3 x_{Supplier1,Customer2} + 1 x_{Supplier1,Customer3} + 2 x_{Supplier1,Customer4} + 3 x_{Supplier1,Customer5} + 2 x_{Supplier1,Customer6}
  + 1 x_{Supplier2,Customer1} + 2 x_{Supplier2,Customer2} + 3 x_{Supplier2,Customer3} + 2 x_{Supplier2,Customer4} + 3 x_{Supplier2,Customer5} + 2 x_{Supplier2,Customer6}
  + 3 x_{Supplier3,Customer1} + 1 x_{Supplier3,Customer2} + 2 x_{Supplier3,Customer3} + 3 x_{Supplier3,Customer4} + 2 x_{Supplier3,Customer5} + 3 x_{Supplier3,Customer6}
  + 2 x_{Supplier4,Customer1} + 3 x_{Supplier4,Customer2} + 2 x_{Supplier4,Customer3} + 1 x_{Supplier4,Customer4} + 3 x_{Supplier4,Customer5} + 4 x_{Supplier4,Customer6}
  + 3 x_{Supplier5,Customer1} + 2 x_{Supplier5,Customer2} + 3 x_{Supplier5,Customer3} + 3 x_{Supplier5,Customer4} + 2 x_{Supplier5,Customer5} + 3 x_{Supplier5,Customer6}

Subject to:

1. Demand satisfaction (for each customer, in source order):
   - x_{Supplier1,Customer1} + x_{Supplier2,Customer1} + x_{Supplier3,Customer1} + x_{Supplier4,Customer1} + x_{Supplier5,Customer1} = 70
   - x_{Supplier1,Customer2} + x_{Supplier2,Customer2} + x_{Supplier3,Customer2} + x_{Supplier4,Customer2} + x_{Supplier5,Customer2} = 80
   - x_{Supplier1,Customer3} + x_{Supplier2,Customer3} + x_{Supplier3,Customer3} + x_{Supplier4,Customer3} + x_{Supplier5,Customer3} = 60
   - x_{Supplier1,Customer4} + x_{Supplier2,Customer4} + x_{Supplier3,Customer4} + x_{Supplier4,Customer4} + x_{Supplier5,Customer4} = 90
   - x_{Supplier1,Customer5} + x_{Supplier2,Customer5} + x_{Supplier3,Customer5} + x_{Supplier4,Customer5} + x_{Supplier5,Customer5} = 85
   - x_{Supplier1,Customer6} + x_{Supplier2,Customer6} + x_{Supplier3,Customer6} + x_{Supplier4,Customer6} + x_{Supplier5,Customer6} = 95

2. Supply capacity (for each supplier, in source order):
   - x_{Supplier1,Customer1} + x_{Supplier1,Customer2} + x_{Supplier1,Customer3} + x_{Supplier1,Customer4} + x_{Supplier1,Customer5} + x_{Supplier1,Customer6} ≤ 200
   - x_{Supplier2,Customer1} + x_{Supplier2,Customer2} + x_{Supplier2,Customer3} + x_{Supplier2,Customer4} + x_{Supplier2,Customer5} + x_{Supplier2,Customer6} ≤ 250
   - x_{Supplier3,Customer1} + x_{Supplier3,Customer2} + x_{Supplier3,Customer3} + x_{Supplier3,Customer4} + x_{Supplier3,Customer5} + x_{Supplier3,Customer6} ≤ 230
   - x_{Supplier4,Customer1} + x_{Supplier4,Customer2} + x_{Supplier4,Customer3} + x_{Supplier4,Customer4} + x_{Supplier4,Customer5} + x_{Supplier4,Customer6} ≤ 220
   - x_{Supplier5,Customer1} + x_{Supplier5,Customer2} + x_{Supplier5,Customer3} + x_{Supplier5,Customer4} + x_{Supplier5,Customer5} + x_{Supplier5,Customer6} ≤ 210

3. Non-negativity:
   - x_{s,c} ≥ 0, ∀ s ∈ S, c ∈ C

This is a complete numerical linear programming formulation for the FreshMart transportation problem, using all identifiers and coefficients from the provided CSVs in source order.