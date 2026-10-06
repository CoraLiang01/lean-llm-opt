Let x_{i,j} be the amount of fresh produce shipped from warehouse (supplier) i to store (customer) j.

Indices:
- i ∈ {Supplier1, Supplier2, Supplier3, Supplier4, Supplier5}
- j ∈ {Customer1, Customer2, Customer3, Customer4, Customer5, Customer6}

Parameters:
- demand_j: daily demand for customer j
    - demand_Customer1 = 70
    - demand_Customer2 = 80
    - demand_Customer3 = 60
    - demand_Customer4 = 90
    - demand_Customer5 = 85
    - demand_Customer6 = 95
- supply_capacity_i: supply capacity of supplier i
    - supply_capacity_Supplier1 = 200
    - supply_capacity_Supplier2 = 250
    - supply_capacity_Supplier3 = 230
    - supply_capacity_Supplier4 = 220
    - supply_capacity_Supplier5 = 210
- c_{i,j}: transportation cost per unit from supplier i to customer j

|            | Customer1 | Customer2 | Customer3 | Customer4 | Customer5 | Customer6 |
|------------|-----------|-----------|-----------|-----------|-----------|-----------|
| Supplier1  |     2     |     3     |     1     |     2     |     3     |     2     |
| Supplier2  |     1     |     2     |     3     |     2     |     3     |     2     |
| Supplier3  |     3     |     1     |     2     |     3     |     2     |     3     |
| Supplier4  |     2     |     3     |     2     |     1     |     3     |     4     |
| Supplier5  |     3     |     2     |     3     |     3     |     2     |     3     |

Variables:
- x_{i,j} ≥ 0, ∀ i, j

Objective:
Minimize total transportation cost:
minimize
2 x_{Supplier1,Customer1} + 3 x_{Supplier1,Customer2} + 1 x_{Supplier1,Customer3} + 2 x_{Supplier1,Customer4} + 3 x_{Supplier1,Customer5} + 2 x_{Supplier1,Customer6}
+ 1 x_{Supplier2,Customer1} + 2 x_{Supplier2,Customer2} + 3 x_{Supplier2,Customer3} + 2 x_{Supplier2,Customer4} + 3 x_{Supplier2,Customer5} + 2 x_{Supplier2,Customer6}
+ 3 x_{Supplier3,Customer1} + 1 x_{Supplier3,Customer2} + 2 x_{Supplier3,Customer3} + 3 x_{Supplier3,Customer4} + 2 x_{Supplier3,Customer5} + 3 x_{Supplier3,Customer6}
+ 2 x_{Supplier4,Customer1} + 3 x_{Supplier4,Customer2} + 2 x_{Supplier4,Customer3} + 1 x_{Supplier4,Customer4} + 3 x_{Supplier4,Customer5} + 4 x_{Supplier4,Customer6}
+ 3 x_{Supplier5,Customer1} + 2 x_{Supplier5,Customer2} + 3 x_{Supplier5,Customer3} + 3 x_{Supplier5,Customer4} + 2 x_{Supplier5,Customer5} + 3 x_{Supplier5,Customer6}

Subject to:

1. Demand satisfaction for each customer:
x_{Supplier1,Customer1} + x_{Supplier2,Customer1} + x_{Supplier3,Customer1} + x_{Supplier4,Customer1} + x_{Supplier5,Customer1} = 70
x_{Supplier1,Customer2} + x_{Supplier2,Customer2} + x_{Supplier3,Customer2} + x_{Supplier4,Customer2} + x_{Supplier5,Customer2} = 80
x_{Supplier1,Customer3} + x_{Supplier2,Customer3} + x_{Supplier3,Customer3} + x_{Supplier4,Customer3} + x_{Supplier5,Customer3} = 60
x_{Supplier1,Customer4} + x_{Supplier2,Customer4} + x_{Supplier3,Customer4} + x_{Supplier4,Customer4} + x_{Supplier5,Customer4} = 90
x_{Supplier1,Customer5} + x_{Supplier2,Customer5} + x_{Supplier3,Customer5} + x_{Supplier4,Customer5} + x_{Supplier5,Customer5} = 85
x_{Supplier1,Customer6} + x_{Supplier2,Customer6} + x_{Supplier3,Customer6} + x_{Supplier4,Customer6} + x_{Supplier5,Customer6} = 95

2. Supply capacity for each supplier:
x_{Supplier1,Customer1} + x_{Supplier1,Customer2} + x_{Supplier1,Customer3} + x_{Supplier1,Customer4} + x_{Supplier1,Customer5} + x_{Supplier1,Customer6} ≤ 200
x_{Supplier2,Customer1} + x_{Supplier2,Customer2} + x_{Supplier2,Customer3} + x_{Supplier2,Customer4} + x_{Supplier2,Customer5} + x_{Supplier2,Customer6} ≤ 250
x_{Supplier3,Customer1} + x_{Supplier3,Customer2} + x_{Supplier3,Customer3} + x_{Supplier3,Customer4} + x_{Supplier3,Customer5} + x_{Supplier3,Customer6} ≤ 230
x_{Supplier4,Customer1} + x_{Supplier4,Customer2} + x_{Supplier4,Customer3} + x_{Supplier4,Customer4} + x_{Supplier4,Customer5} + x_{Supplier4,Customer6} ≤ 220
x_{Supplier5,Customer1} + x_{Supplier5,Customer2} + x_{Supplier5,Customer3} + x_{Supplier5,Customer4} + x_{Supplier5,Customer5} + x_{Supplier5,Customer6} ≤ 210

3. Non-negativity:
x_{i,j} ≥ 0 for all i, j

This is the complete numerical formulation of the transportation problem for FreshMart as described.