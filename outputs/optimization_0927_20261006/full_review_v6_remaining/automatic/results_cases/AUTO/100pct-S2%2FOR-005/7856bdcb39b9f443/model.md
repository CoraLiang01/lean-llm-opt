Let $x_i$ be the number of units of bread type $i$ to order each day, where $i$ indexes the following breads in the order given:

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

Let $p_i$ be the expected profit per unit of bread $i$ ("item_value"), and $a_i$ be the storage space required per unit of bread $i$ ("resource_requirement"). The total available storage capacity is $180$ ("resource_capacity").

The mathematical model is:

$$
\begin{align*}
\text{Maximize} \quad & 888x_1 + 134x_2 + 129x_3 + 370x_4 + 921x_5 + 765x_6 + 154x_7 + 837x_8 + 584x_9 + 365x_{10} \\[2ex]
\text{subject to} \quad & 4x_1 + 2x_2 + 4x_3 + 3x_4 + 2x_5 + 1x_6 + 2x_7 + 1x_8 + 3x_9 + 3x_{10} \leq 180 \\[2ex]
& x_i \in \mathbb{Z}_{\geq 0} \quad \text{for } i = 1, \ldots, 10
\end{align*}
$$

Where:

- $x_1$: Baguette
- $x_2$: Croissant
- $x_3$: Sourdough
- $x_4$: Rye Bread
- $x_5$: Brioche
- $x_6$: Focaccia
- $x_7$: Ciabatta
- $x_8$: Pita
- $x_9$: Bagel
- $x_{10}$: English Muffin

All variables are nonnegative integers. The objective is to maximize total expected profit, subject to the daily storage capacity constraint.