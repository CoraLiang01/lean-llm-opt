Let $x_i$ be the number of units of bread type $i$ to order each day, where $i$ indexes the following bread types in the order given:

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

Let $p_i$ be the expected profit per unit of bread type $i$, and $a_i$ be the storage space required per unit of bread type $i$.

The total storage capacity available per day is $180$.

The model is:

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

Where:

\[
\begin{array}{llll}
\text{Bread Type} & p_i \ (\text{item\_value}) & a_i \ (\text{resource\_requirement}) \\
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

Storage capacity: $180$ units. 

All $x_i$ are nonnegative integers.