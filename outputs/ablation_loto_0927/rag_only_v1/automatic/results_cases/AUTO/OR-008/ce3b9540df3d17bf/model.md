Let x_{i,j} denote the amount of fresh produce shipped from warehouse (supplier) i to store (customer) j.

Sets:
- Suppliers (warehouses): i ∈ {Supplier1, Supplier2, Supplier3, Supplier4, Supplier5}
- Customers (stores): j ∈ {Customer1, Customer2, Customer3, Customer4, Customer5, Customer6}

Parameters:
- demand_j: daily demand for customer j
    - demand_1 = 70
    - demand_2 = 80
    - demand_3 = 60
    - demand_4 = 90
    - demand_5 = 85
    - demand_6 = 95
- supply_capacity_i: supply capacity for supplier i
    - supply_capacity_1 = 200
    - supply_capacity_2 = 250
    - supply_capacity_3 = 230
    - supply_capacity_4 = 220
    - supply_capacity_5 = 210
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
2 x_{1,1} + 3 x_{1,2} + 1 x_{1,3} + 2 x_{1,4} + 3 x_{1,5} + 2 x_{1,6}
+ 1 x_{2,1} + 2 x_{2,2} + 3 x_{2,3} + 2 x_{2,4} + 3 x_{2,5} + 2 x_{2,6}
+ 3 x_{3,1} + 1 x_{3,2} + 2 x_{3,3} + 3 x_{3,4} + 2 x_{3,5} + 3 x_{3,6}
+ 2 x_{4,1} + 3 x_{4,2} + 2 x_{4,3} + 1 x_{4,4} + 3 x_{4,5} + 4 x_{4,6}
+ 3 x_{5,1} + 2 x_{5,2} + 3 x_{5,3} + 3 x_{5,4} + 2 x_{5,5} + 3 x_{5,6}

Subject to:

1. Demand satisfaction for each customer:
x_{1,1} + x_{2,1} + x_{3,1} + x_{4,1} + x_{5,1} = 70
x_{1,2} + x_{2,2} + x_{3,2} + x_{4,2} + x_{5,2} = 80
x_{1,3} + x_{2,3} + x_{3,3} + x_{4,3} + x_{5,3} = 60
x_{1,4} + x_{2,4} + x_{3,4} + x_{4,4} + x_{5,4} = 90
x_{1,5} + x_{2,5} + x_{3,5} + x_{4,5} + x_{5,5} = 85
x_{1,6} + x_{2,6} + x_{3,6} + x_{4,6} + x_{5,6} = 95

2. Supply capacity for each supplier:
x_{1,1} + x_{1,2} + x_{1,3} + x_{1,4} + x_{1,5} + x_{1,6} ≤ 200
x_{2,1} + x_{2,2} + x_{2,3} + x_{2,4} + x_{2,5} + x_{2,6} ≤ 250
x_{3,1} + x_{3,2} + x_{3,3} + x_{3,4} + x_{3,5} + x_{3,6} ≤ 230
x_{4,1} + x_{4,2} + x_{4,3} + x_{4,4} + x_{4,5} + x_{4,6} ≤ 220
x_{5,1} + x_{5,2} + x_{5,3} + x_{5,4} + x_{5,5} + x_{5,6} ≤ 210

3. Non-negativity:
x_{i,j} ≥ 0, ∀ i ∈ {1,...,5}, j ∈ {1,...,6}

This is a complete numerical linear programming formulation for the FreshMart transportation problem.