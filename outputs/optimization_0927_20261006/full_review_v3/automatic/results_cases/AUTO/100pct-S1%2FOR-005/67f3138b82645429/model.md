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

The parameters for each bread type $i$ are:

| item_name        | item_value | resource_requirement |
|------------------|------------|---------------------|
| Baguette         | 888        | 4                   |
| Croissant        | 134        | 2                   |
| Sourdough        | 129        | 4                   |
| Rye Bread        | 370        | 3                   |
| Brioche          | 921        | 2                   |
| Focaccia         | 765        | 1                   |
| Ciabatta         | 154        | 2                   |
| Pita             | 837        | 1                   |
| Bagel            | 584        | 3                   |
| English Muffin   | 365        | 3                   |

The total storage capacity is 180.

The mathematical model is:

Objective:
$$
\max\ 888x_1 + 134x_2 + 129x_3 + 370x_4 + 921x_5 + 765x_6 + 154x_7 + 837x_8 + 584x_9 + 365x_{10}
$$

Subject to:
$$
4x_1 + 2x_2 + 4x_3 + 3x_4 + 2x_5 + 1x_6 + 2x_7 + 1x_8 + 3x_9 + 3x_{10} \leq 180
$$

$$
x_i \in \mathbb{Z}_{\geq 0},\quad \forall i = 1,\ldots,10
$$

Where:
- $x_i$ = number of units of bread type $i$ to order each day (integer, nonnegative)
- The coefficients in the objective are the expected profits per unit for each bread type.
- The coefficients in the constraint are the storage requirements per unit for each bread type.
- The right-hand side of the constraint is the total storage capacity.