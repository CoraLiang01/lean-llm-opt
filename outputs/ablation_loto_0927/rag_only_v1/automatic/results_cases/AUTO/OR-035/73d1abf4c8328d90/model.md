Let \( x_i \) be the number of units of bread type \( i \) to order each day, where \( i \) indexes the following bread types in the order given:

1. Baguette
2. Croissant
3. Sourdough
4. Rye Bread
5. Brioche
6. Focaccia
7. Ciabatta
8. Pita
9. Bagel
10. English Muffin

All \( x_i \) are nonnegative integers.

Parameters (from products.csv and capacity.csv):

| i  | ProductName      | Value (profit/unit) | Weight (storage/unit) |
|----|------------------|--------------------|-----------------------|
| 1  | Baguette         | 888                | 4                     |
| 2  | Croissant        | 134                | 2                     |
| 3  | Sourdough        | 129                | 4                     |
| 4  | Rye Bread        | 370                | 3                     |
| 5  | Brioche          | 921                | 2                     |
| 6  | Focaccia         | 765                | 1                     |
| 7  | Ciabatta         | 154                | 2                     |
| 8  | Pita             | 837                | 1                     |
| 9  | Bagel            | 584                | 3                     |
| 10 | English Muffin   | 365                | 3                     |

Total storage capacity: 180

Mathematical Model:

Decision variables:
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \text{for } i = 1, \ldots, 10
\]

Objective:
\[
\max \; 888x_1 + 134x_2 + 129x_3 + 370x_4 + 921x_5 + 765x_6 + 154x_7 + 837x_8 + 584x_9 + 365x_{10}
\]

Subject to (storage capacity constraint):
\[
4x_1 + 2x_2 + 4x_3 + 3x_4 + 2x_5 + 1x_6 + 2x_7 + 1x_8 + 3x_9 + 3x_{10} \leq 180
\]

\[
x_i \geq 0 \text{ and integer} \quad \forall i = 1, \ldots, 10
\]

Where:
- \( x_i \) = number of units of bread type \( i \) to order each day
- The coefficients in the objective are the expected profits per unit for each bread type
- The coefficients in the constraint are the storage weights per unit for each bread type
- The right-hand side of the constraint is the total storage capacity

This model maximizes the total expected profit from daily bread orders, subject to the bakery's storage limit.