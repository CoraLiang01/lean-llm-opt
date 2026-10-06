Let \( x_{ij} \) denote the amount of fresh produce shipped from warehouse (supplier) \( i \) to store (customer) \( j \).

Indices:
- \( i \in \{\text{Supplier1}, \text{Supplier2}, \text{Supplier3}, \text{Supplier4}, \text{Supplier5}\} \)
- \( j \in \{\text{Customer1}, \text{Customer2}, \text{Customer3}, \text{Customer4}, \text{Customer5}, \text{Customer6}\} \)

Parameters:
- Customer demands:
  - \( d_1 = 70 \) (Customer1)
  - \( d_2 = 80 \) (Customer2)
  - \( d_3 = 60 \) (Customer3)
  - \( d_4 = 90 \) (Customer4)
  - \( d_5 = 85 \) (Customer5)
  - \( d_6 = 95 \) (Customer6)
- Supplier capacities:
  - \( s_1 = 200 \) (Supplier1)
  - \( s_2 = 250 \) (Supplier2)
  - \( s_3 = 230 \) (Supplier3)
  - \( s_4 = 220 \) (Supplier4)
  - \( s_5 = 210 \) (Supplier5)
- Transportation costs \( c_{ij} \):

|            | Customer1 | Customer2 | Customer3 | Customer4 | Customer5 | Customer6 |
|------------|-----------|-----------|-----------|-----------|-----------|-----------|
| Supplier1  |     2     |     3     |     1     |     2     |     3     |     2     |
| Supplier2  |     1     |     2     |     3     |     2     |     3     |     2     |
| Supplier3  |     3     |     1     |     2     |     3     |     2     |     3     |
| Supplier4  |     2     |     3     |     2     |     1     |     3     |     4     |
| Supplier5  |     3     |     2     |     3     |     3     |     2     |     3     |

Variables:
- \( x_{ij} \geq 0 \) for all \( i, j \)

Objective:
\[
\min \sum_{i=1}^{5} \sum_{j=1}^{6} c_{ij} x_{ij}
\]
That is,
\[
\min \Big[
2x_{1,1} + 3x_{1,2} + 1x_{1,3} + 2x_{1,4} + 3x_{1,5} + 2x_{1,6} +
1x_{2,1} + 2x_{2,2} + 3x_{2,3} + 2x_{2,4} + 3x_{2,5} + 2x_{2,6} +
3x_{3,1} + 1x_{3,2} + 2x_{3,3} + 3x_{3,4} + 2x_{3,5} + 3x_{3,6} +
2x_{4,1} + 3x_{4,2} + 2x_{4,3} + 1x_{4,4} + 3x_{4,5} + 4x_{4,6} +
3x_{5,1} + 2x_{5,2} + 3x_{5,3} + 3x_{5,4} + 2x_{5,5} + 3x_{5,6}
\Big]
\]

Subject to:

1. Demand satisfaction for each customer:
\[
\begin{align*}
x_{1,1} + x_{2,1} + x_{3,1} + x_{4,1} + x_{5,1} &= 70 \\
x_{1,2} + x_{2,2} + x_{3,2} + x_{4,2} + x_{5,2} &= 80 \\
x_{1,3} + x_{2,3} + x_{3,3} + x_{4,3} + x_{5,3} &= 60 \\
x_{1,4} + x_{2,4} + x_{3,4} + x_{4,4} + x_{5,4} &= 90 \\
x_{1,5} + x_{2,5} + x_{3,5} + x_{4,5} + x_{5,5} &= 85 \\
x_{1,6} + x_{2,6} + x_{3,6} + x_{4,6} + x_{5,6} &= 95 \\
\end{align*}
\]

2. Supply capacity for each supplier:
\[
\begin{align*}
x_{1,1} + x_{1,2} + x_{1,3} + x_{1,4} + x_{1,5} + x_{1,6} &\leq 200 \\
x_{2,1} + x_{2,2} + x_{2,3} + x_{2,4} + x_{2,5} + x_{2,6} &\leq 250 \\
x_{3,1} + x_{3,2} + x_{3,3} + x_{3,4} + x_{3,5} + x_{3,6} &\leq 230 \\
x_{4,1} + x_{4,2} + x_{4,3} + x_{4,4} + x_{4,5} + x_{4,6} &\leq 220 \\
x_{5,1} + x_{5,2} + x_{5,3} + x_{5,4} + x_{5,5} + x_{5,6} &\leq 210 \\
\end{align*}
\]

3. Non-negativity:
\[
x_{ij} \geq 0 \quad \forall i, j
\]

This is the complete numerical formulation of the transportation problem for FreshMart, using all identifiers and coefficients as provided in the source data.