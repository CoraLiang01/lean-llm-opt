Let the decision variable \( x_i \) represent the number of units of bread type \( i \) to order each day, where \( x_i \) is a nonnegative integer for each bread type \( i \).

Define the following indices and parameters from the data:

Let the set of bread types (in source order) be:
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

Let:
- \( v_i \) = expected profit per unit of bread type \( i \)
- \( w_i \) = storage weight per unit of bread type \( i \)
- \( C \) = total storage capacity

From the data:
\[
\begin{array}{llll}
\text{Index} & \text{ProductName} & v_i & w_i \\
1 & \text{Baguette} & 888 & 4 \\
2 & \text{Croissant} & 134 & 2 \\
3 & \text{Sourdough} & 129 & 4 \\
4 & \text{Rye Bread} & 370 & 3 \\
5 & \text{Brioche} & 921 & 2 \\
6 & \text{Focaccia} & 765 & 1 \\
7 & \text{Ciabatta} & 154 & 2 \\
8 & \text{Pita} & 837 & 1 \\
9 & \text{Bagel} & 584 & 3 \\
10 & \text{English Muffin} & 365 & 3 \\
\end{array}
\]
Total storage capacity: \( C = 180 \)

Model:

\[
\begin{align*}
\text{Maximize} \quad & 888x_1 + 134x_2 + 129x_3 + 370x_4 + 921x_5 + 765x_6 + 154x_7 + 837x_8 + 584x_9 + 365x_{10} \\
\text{subject to} \quad & 4x_1 + 2x_2 + 4x_3 + 3x_4 + 2x_5 + 1x_6 + 2x_7 + 1x_8 + 3x_9 + 3x_{10} \leq 180 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \text{for } i = 1, \ldots, 10
\end{align*}
\]

Where:
- \( x_1 \): units of Baguette
- \( x_2 \): units of Croissant
- \( x_3 \): units of Sourdough
- \( x_4 \): units of Rye Bread
- \( x_5 \): units of Brioche
- \( x_6 \): units of Focaccia
- \( x_7 \): units of Ciabatta
- \( x_8 \): units of Pita
- \( x_9 \): units of Bagel
- \( x_{10} \): units of English Muffin

All variables are nonnegative integers.

This model maximizes the total expected profit from bread sales, subject to the bakery's daily storage capacity.