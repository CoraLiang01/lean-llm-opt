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

Let $p_i$ be the expected profit per unit of bread type $i$ (from "item_value"), and $a_i$ be the storage space required per unit of bread type $i$ (from "resource_requirement"). The total available storage capacity is $180$ (from "resource_capacity" in "capacity.csv").

The complete mathematical model is:

Maximize total expected profit:
$$
\max \quad 888 x_1 + 134 x_2 + 129 x_3 + 370 x_4 + 921 x_5 + 765 x_6 + 154 x_7 + 837 x_8 + 584 x_9 + 365 x_{10}
$$

Subject to the storage capacity constraint:
$$
4 x_1 + 2 x_2 + 4 x_3 + 3 x_4 + 2 x_5 + 1 x_6 + 2 x_7 + 1 x_8 + 3 x_9 + 3 x_{10} \leq 180
$$

Integer and nonnegativity constraints:
$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 10
$$

Where:

| $i$ | item_name         | $p_i$ (item_value) | $a_i$ (resource_requirement) |
|-----|-------------------|--------------------|------------------------------|
| 1   | Baguette          | 888                | 4                            |
| 2   | Croissant         | 134                | 2                            |
| 3   | Sourdough         | 129                | 4                            |
| 4   | Rye Bread         | 370                | 3                            |
| 5   | Brioche           | 921                | 2                            |
| 6   | Focaccia          | 765                | 1                            |
| 7   | Ciabatta          | 154                | 2                            |
| 8   | Pita              | 837                | 1                            |
| 9   | Bagel             | 584                | 3                            |
| 10  | English Muffin    | 365                | 3                            |

All variables $x_i$ are nonnegative integers.