Sets:
- Let S = {Supplier1, Supplier2, Supplier3, Supplier4, Supplier5} be the set of warehouses (suppliers).
- Let C = {Customer1, Customer2, Customer3, Customer4, Customer5, Customer6} be the set of retail stores (customers).

Parameters:
- Demand for each customer:
    - d_Customer1 = 70
    - d_Customer2 = 80
    - d_Customer3 = 60
    - d_Customer4 = 90
    - d_Customer5 = 85
    - d_Customer6 = 95
- Supply capacity for each supplier:
    - cap_Supplier1 = 200
    - cap_Supplier2 = 250
    - cap_Supplier3 = 230
    - cap_Supplier4 = 220
    - cap_Supplier5 = 210
- Transportation cost per unit from each supplier to each customer (c_{s,c}):

|            | Customer1 | Customer2 | Customer3 | Customer4 | Customer5 | Customer6 |
|------------|-----------|-----------|-----------|-----------|-----------|-----------|
| Supplier1  |     2     |     3     |     1     |     2     |     3     |     2     |
| Supplier2  |     1     |     2     |     3     |     2     |     3     |     2     |
| Supplier3  |     3     |     1     |     2     |     3     |     2     |     3     |
| Supplier4  |     2     |     3     |     2     |     1     |     3     |     4     |
| Supplier5  |     3     |     2     |     3     |     3     |     2     |     3     |

Decision Variables:
- x_{s,c} ≥ 0: Amount of fresh produce shipped from supplier s ∈ S to customer c ∈ C.

Objective:
Minimize total transportation cost:
\[
\text{Minimize} \quad Z = \sum_{s \in S} \sum_{c \in C} c_{s,c} \cdot x_{s,c}
\]
That is,
\[
\text{Minimize} \quad
2x_{Supplier1,Customer1} + 3x_{Supplier1,Customer2} + 1x_{Supplier1,Customer3} + 2x_{Supplier1,Customer4} + 3x_{Supplier1,Customer5} + 2x_{Supplier1,Customer6} \\
+ 1x_{Supplier2,Customer1} + 2x_{Supplier2,Customer2} + 3x_{Supplier2,Customer3} + 2x_{Supplier2,Customer4} + 3x_{Supplier2,Customer5} + 2x_{Supplier2,Customer6} \\
+ 3x_{Supplier3,Customer1} + 1x_{Supplier3,Customer2} + 2x_{Supplier3,Customer3} + 3x_{Supplier3,Customer4} + 2x_{Supplier3,Customer5} + 3x_{Supplier3,Customer6} \\
+ 2x_{Supplier4,Customer1} + 3x_{Supplier4,Customer2} + 2x_{Supplier4,Customer3} + 1x_{Supplier4,Customer4} + 3x_{Supplier4,Customer5} + 4x_{Supplier4,Customer6} \\
+ 3x_{Supplier5,Customer1} + 2x_{Supplier5,Customer2} + 3x_{Supplier5,Customer3} + 3x_{Supplier5,Customer4} + 2x_{Supplier5,Customer5} + 3x_{Supplier5,Customer6}
\]

Subject to:

1. Demand satisfaction for each customer:
\[
\sum_{s \in S} x_{s,c} = d_c \quad \forall c \in C
\]
Explicitly:
\[
x_{Supplier1,Customer1} + x_{Supplier2,Customer1} + x_{Supplier3,Customer1} + x_{Supplier4,Customer1} + x_{Supplier5,Customer1} = 70 \\
x_{Supplier1,Customer2} + x_{Supplier2,Customer2} + x_{Supplier3,Customer2} + x_{Supplier4,Customer2} + x_{Supplier5,Customer2} = 80 \\
x_{Supplier1,Customer3} + x_{Supplier2,Customer3} + x_{Supplier3,Customer3} + x_{Supplier4,Customer3} + x_{Supplier5,Customer3} = 60 \\
x_{Supplier1,Customer4} + x_{Supplier2,Customer4} + x_{Supplier3,Customer4} + x_{Supplier4,Customer4} + x_{Supplier5,Customer4} = 90 \\
x_{Supplier1,Customer5} + x_{Supplier2,Customer5} + x_{Supplier3,Customer5} + x_{Supplier4,Customer5} + x_{Supplier5,Customer5} = 85 \\
x_{Supplier1,Customer6} + x_{Supplier2,Customer6} + x_{Supplier3,Customer6} + x_{Supplier4,Customer6} + x_{Supplier5,Customer6} = 95
\]

2. Supply capacity for each supplier:
\[
\sum_{c \in C} x_{s,c} \leq cap_s \quad \forall s \in S
\]
Explicitly:
\[
x_{Supplier1,Customer1} + x_{Supplier1,Customer2} + x_{Supplier1,Customer3} + x_{Supplier1,Customer4} + x_{Supplier1,Customer5} + x_{Supplier1,Customer6} \leq 200 \\
x_{Supplier2,Customer1} + x_{Supplier2,Customer2} + x_{Supplier2,Customer3} + x_{Supplier2,Customer4} + x_{Supplier2,Customer5} + x_{Supplier2,Customer6} \leq 250 \\
x_{Supplier3,Customer1} + x_{Supplier3,Customer2} + x_{Supplier3,Customer3} + x_{Supplier3,Customer4} + x_{Supplier3,Customer5} + x_{Supplier3,Customer6} \leq 230 \\
x_{Supplier4,Customer1} + x_{Supplier4,Customer2} + x_{Supplier4,Customer3} + x_{Supplier4,Customer4} + x_{Supplier4,Customer5} + x_{Supplier4,Customer6} \leq 220 \\
x_{Supplier5,Customer1} + x_{Supplier5,Customer2} + x_{Supplier5,Customer3} + x_{Supplier5,Customer4} + x_{Supplier5,Customer5} + x_{Supplier5,Customer6} \leq 210
\]

3. Non-negativity:
\[
x_{s,c} \geq 0 \quad \forall s \in S, c \in C
\]

This is a complete linear programming formulation for the FreshMart transportation problem as described.