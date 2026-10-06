Let x_{ij} denote the amount of fresh produce shipped from Supplier i to Customer j, where i ∈ {1,2,3,4,5} and j ∈ {1,2,3,4,5,6}.

Parameters (from CSV evidence):

Customer demands:
- Customer1: 70
- Customer2: 80
- Customer3: 60
- Customer4: 90
- Customer5: 85
- Customer6: 95

Supplier capacities:
- Supplier1: 200
- Supplier2: 250
- Supplier3: 230
- Supplier4: 220
- Supplier5: 210

Transportation costs per unit (c_{ij}):

|            | Customer1 | Customer2 | Customer3 | Customer4 | Customer5 | Customer6 |
|------------|-----------|-----------|-----------|-----------|-----------|-----------|
| Supplier1  |     2     |     3     |     1     |     2     |     3     |     2     |
| Supplier2  |     1     |     2     |     3     |     2     |     3     |     2     |
| Supplier3  |     3     |     1     |     2     |     3     |     2     |     3     |
| Supplier4  |     2     |     3     |     2     |     1     |     3     |     4     |
| Supplier5  |     3     |     2     |     3     |     3     |     2     |     3     |

Decision variables:
x_{ij} ≥ 0, continuous, for all i ∈ {1,2,3,4,5}, j ∈ {1,2,3,4,5,6}

Objective:
Minimize total transportation cost:
minimize
2x_{1,1} + 3x_{1,2} + 1x_{1,3} + 2x_{1,4} + 3x_{1,5} + 2x_{1,6}
+ 1x_{2,1} + 2x_{2,2} + 3x_{2,3} + 2x_{2,4} + 3x_{2,5} + 2x_{2,6}
+ 3x_{3,1} + 1x_{3,2} + 2x_{3,3} + 3x_{3,4} + 2x_{3,5} + 3x_{3,6}
+ 2x_{4,1} + 3x_{4,2} + 2x_{4,3} + 1x_{4,4} + 3x_{4,5} + 4x_{4,6}
+ 3x_{5,1} + 2x_{5,2} + 3x_{5,3} + 3x_{5,4} + 2x_{5,5} + 3x_{5,6}

Subject to:

Demand satisfaction (for each customer):
x_{1,1} + x_{2,1} + x_{3,1} + x_{4,1} + x_{5,1} = 70      (Customer1)
x_{1,2} + x_{2,2} + x_{3,2} + x_{4,2} + x_{5,2} = 80      (Customer2)
x_{1,3} + x_{2,3} + x_{3,3} + x_{4,3} + x_{5,3} = 60      (Customer3)
x_{1,4} + x_{2,4} + x_{3,4} + x_{4,4} + x_{5,4} = 90      (Customer4)
x_{1,5} + x_{2,5} + x_{3,5} + x_{4,5} + x_{5,5} = 85      (Customer5)
x_{1,6} + x_{2,6} + x_{3,6} + x_{4,6} + x_{5,6} = 95      (Customer6)

Supply capacity (for each supplier):
x_{1,1} + x_{1,2} + x_{1,3} + x_{1,4} + x_{1,5} + x_{1,6} ≤ 200   (Supplier1)
x_{2,1} + x_{2,2} + x_{2,3} + x_{2,4} + x_{2,5} + x_{2,6} ≤ 250   (Supplier2)
x_{3,1} + x_{3,2} + x_{3,3} + x_{3,4} + x_{3,5} + x_{3,6} ≤ 230   (Supplier3)
x_{4,1} + x_{4,2} + x_{4,3} + x_{4,4} + x_{4,5} + x_{4,6} ≤ 220   (Supplier4)
x_{5,1} + x_{5,2} + x_{5,3} + x_{5,4} + x_{5,5} + x_{5,6} ≤ 210   (Supplier5)

Non-negativity:
x_{ij} ≥ 0 for all i ∈ {1,2,3,4,5}, j ∈ {1,2,3,4,5,6}

This is the complete numerical formulation of the FreshMart transportation optimization problem, preserving all identifiers, coefficients, and source order from the CSV evidence.