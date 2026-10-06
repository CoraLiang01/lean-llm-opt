Let:
- S = {Supplier1, Supplier2, Supplier3, Supplier4, Supplier5} (warehouses)
- C = {Customer1, Customer2, Customer3, Customer4, Customer5, Customer6} (stores)
- x_{i,j} = amount of produce shipped from Supplier i to Customer j (decision variables, continuous, x_{i,j} ≥ 0)

Parameters (from CSVs, in source order):

Customer demands:
- demand_{Customer1} = 70
- demand_{Customer2} = 80
- demand_{Customer3} = 60
- demand_{Customer4} = 90
- demand_{Customer5} = 85
- demand_{Customer6} = 95

Supplier capacities:
- supply_{Supplier1} = 200
- supply_{Supplier2} = 250
- supply_{Supplier3} = 230
- supply_{Supplier4} = 220
- supply_{Supplier5} = 210

Transportation costs per unit (c_{i,j}):

|            | Customer1 | Customer2 | Customer3 | Customer4 | Customer5 | Customer6 |
|------------|-----------|-----------|-----------|-----------|-----------|-----------|
| Supplier1  |     2     |     3     |     1     |     2     |     3     |     2     |
| Supplier2  |     1     |     2     |     3     |     2     |     3     |     2     |
| Supplier3  |     3     |     1     |     2     |     3     |     2     |     3     |
| Supplier4  |     2     |     3     |     2     |     1     |     3     |     4     |
| Supplier5  |     3     |     2     |     3     |     3     |     2     |     3     |

Mathematical Model:

Variables:
x_{i,j} ≥ 0 for all i ∈ S, j ∈ C

Objective:
Minimize total transportation cost:
minimize
 2x_{Supplier1,Customer1} + 3x_{Supplier1,Customer2} + 1x_{Supplier1,Customer3} + 2x_{Supplier1,Customer4} + 3x_{Supplier1,Customer5} + 2x_{Supplier1,Customer6}
+ 1x_{Supplier2,Customer1} + 2x_{Supplier2,Customer2} + 3x_{Supplier2,Customer3} + 2x_{Supplier2,Customer4} + 3x_{Supplier2,Customer5} + 2x_{Supplier2,Customer6}
+ 3x_{Supplier3,Customer1} + 1x_{Supplier3,Customer2} + 2x_{Supplier3,Customer3} + 3x_{Supplier3,Customer4} + 2x_{Supplier3,Customer5} + 3x_{Supplier3,Customer6}
+ 2x_{Supplier4,Customer1} + 3x_{Supplier4,Customer2} + 2x_{Supplier4,Customer3} + 1x_{Supplier4,Customer4} + 3x_{Supplier4,Customer5} + 4x_{Supplier4,Customer6}
+ 3x_{Supplier5,Customer1} + 2x_{Supplier5,Customer2} + 3x_{Supplier5,Customer3} + 3x_{Supplier5,Customer4} + 2x_{Supplier5,Customer5} + 3x_{Supplier5,Customer6}

Subject to:

1. Demand satisfaction (for each customer):
 x_{Supplier1,Customer1} + x_{Supplier2,Customer1} + x_{Supplier3,Customer1} + x_{Supplier4,Customer1} + x_{Supplier5,Customer1} = 70
 x_{Supplier1,Customer2} + x_{Supplier2,Customer2} + x_{Supplier3,Customer2} + x_{Supplier4,Customer2} + x_{Supplier5,Customer2} = 80
 x_{Supplier1,Customer3} + x_{Supplier2,Customer3} + x_{Supplier3,Customer3} + x_{Supplier4,Customer3} + x_{Supplier5,Customer3} = 60
 x_{Supplier1,Customer4} + x_{Supplier2,Customer4} + x_{Supplier3,Customer4} + x_{Supplier4,Customer4} + x_{Supplier5,Customer4} = 90
 x_{Supplier1,Customer5} + x_{Supplier2,Customer5} + x_{Supplier3,Customer5} + x_{Supplier4,Customer5} + x_{Supplier5,Customer5} = 85
 x_{Supplier1,Customer6} + x_{Supplier2,Customer6} + x_{Supplier3,Customer6} + x_{Supplier4,Customer6} + x_{Supplier5,Customer6} = 95

2. Supply capacity (for each supplier):
 x_{Supplier1,Customer1} + x_{Supplier1,Customer2} + x_{Supplier1,Customer3} + x_{Supplier1,Customer4} + x_{Supplier1,Customer5} + x_{Supplier1,Customer6} ≤ 200
 x_{Supplier2,Customer1} + x_{Supplier2,Customer2} + x_{Supplier2,Customer3} + x_{Supplier2,Customer4} + x_{Supplier2,Customer5} + x_{Supplier2,Customer6} ≤ 250
 x_{Supplier3,Customer1} + x_{Supplier3,Customer2} + x_{Supplier3,Customer3} + x_{Supplier3,Customer4} + x_{Supplier3,Customer5} + x_{Supplier3,Customer6} ≤ 230
 x_{Supplier4,Customer1} + x_{Supplier4,Customer2} + x_{Supplier4,Customer3} + x_{Supplier4,Customer4} + x_{Supplier4,Customer5} + x_{Supplier4,Customer6} ≤ 220
 x_{Supplier5,Customer1} + x_{Supplier5,Customer2} + x_{Supplier5,Customer3} + x_{Supplier5,Customer4} + x_{Supplier5,Customer5} + x_{Supplier5,Customer6} ≤ 210

3. Non-negativity:
 x_{i,j} ≥ 0 for all i ∈ S, j ∈ C

This is the complete numerical formulation of the FreshMart transportation optimization problem, preserving all source-ordered identifiers and coefficients.