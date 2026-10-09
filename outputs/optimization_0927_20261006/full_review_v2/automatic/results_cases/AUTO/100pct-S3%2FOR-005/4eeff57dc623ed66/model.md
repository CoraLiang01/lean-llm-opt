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

The total available storage capacity is $180$ units.

The data is as follows:

| $i$ | item_name        | $p_i$ (item_value) | $a_i$ (resource_requirement) |
|-----|------------------|--------------------|------------------------------|
| 1   | Baguette         | 888                | 4                            |
| 2   | Croissant        | 134                | 2                            |
| 3   | Sourdough        | 129                | 4                            |
| 4   | Rye Bread        | 370                | 3                            |
| 5   | Brioche          | 921                | 2                            |
| 6   | Focaccia         | 765                | 1                            |
| 7   | Ciabatta         | 154                | 2                            |
| 8   | Pita             | 837                | 1                            |
| 9   | Bagel            | 584                | 3                            |
| 10  | English Muffin   | 365                | 3                            |

The mathematical model is:

$$
\begin{align*}
\text{Maximize} \quad & 888x_1 + 134x_2 + 129x_3 + 370x_4 + 921x_5 + 765x_6 + 154x_7 + 837x_8 + 584x_9 + 365x_{10} \\
\text{subject to} \quad & 4x_1 + 2x_2 + 4x_3 + 3x_4 + 2x_5 + 1x_6 + 2x_7 + 1x_8 + 3x_9 + 3x_{10} \leq 180 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 10
\end{align*}
$$

Where:
- $x_i$ = number of units of bread type $i$ to order each day (integer, $\geq 0$)
- $p_i$ = expected profit per unit of bread type $i$ (see table above)
- $a_i$ = storage space required per unit of bread type $i$ (see table above)
- The total storage used cannot exceed $180$ units.