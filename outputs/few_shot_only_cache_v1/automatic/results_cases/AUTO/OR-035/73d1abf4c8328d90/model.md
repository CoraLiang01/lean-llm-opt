Let $x_i$ be the number of units of bread type $i$ to order each day, where $i$ indexes the following products in the order given:

\[
\begin{array}{ll}
1: & \text{Baguette} \\
2: & \text{Croissant} \\
3: & \text{Sourdough} \\
4: & \text{Rye Bread} \\
5: & \text{Brioche} \\
6: & \text{Focaccia} \\
7: & \text{Ciabatta} \\
8: & \text{Pita} \\
9: & \text{Bagel} \\
10: & \text{English Muffin} \\
\end{array}
\]

Parameters:
\[
\begin{array}{lll}
\text{Product} & \text{Value } (v_i) & \text{Weight } (w_i) \\
\hline
\text{Baguette} & 888 & 4 \\
\text{Croissant} & 134 & 2 \\
\text{Sourdough} & 129 & 4 \\
\text{Rye Bread} & 370 & 3 \\
\text{Brioche} & 921 & 2 \\
\text{Focaccia} & 765 & 1 \\
\text{Ciabatta} & 154 & 2 \\
\text{Pita} & 837 & 1 \\
\text{Bagel} & 584 & 3 \\
\text{English Muffin} & 365 & 3 \\
\end{array}
\]
Total storage capacity: $C = 180$

Objective:
\[
\max \; 888x_1 + 134x_2 + 129x_3 + 370x_4 + 921x_5 + 765x_6 + 154x_7 + 837x_8 + 584x_9 + 365x_{10}
\]

Subject to:
\[
4x_1 + 2x_2 + 4x_3 + 3x_4 + 2x_5 + 1x_6 + 2x_7 + 1x_8 + 3x_9 + 3x_{10} \leq 180
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 10
\]