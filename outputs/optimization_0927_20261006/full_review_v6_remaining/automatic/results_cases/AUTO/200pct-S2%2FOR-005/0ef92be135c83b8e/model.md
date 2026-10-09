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

Parameters (from the data):

| $i$ | item_name         | item_value | resource_requirement |
|-----|-------------------|------------|----------------------|
| 1   | Baguette          | 888        | 4                    |
| 2   | Croissant         | 134        | 2                    |
| 3   | Sourdough         | 129        | 4                    |
| 4   | Rye Bread         | 370        | 3                    |
| 5   | Brioche           | 921        | 2                    |
| 6   | Focaccia          | 765        | 1                    |
| 7   | Ciabatta          | 154        | 2                    |
| 8   | Pita              | 837        | 1                    |
| 9   | Bagel             | 584        | 3                    |
| 10  | English Muffin    | 365        | 3                    |

Total storage capacity: $180$

Mathematical Model:

Objective:
$$
\max \; 888x_1 + 134x_2 + 129x_3 + 370x_4 + 921x_5 + 765x_6 + 154x_7 + 837x_8 + 584x_9 + 365x_{10}
$$

Subject to:
$$
4x_1 + 2x_2 + 4x_3 + 3x_4 + 2x_5 + 1x_6 + 2x_7 + 1x_8 + 3x_9 + 3x_{10} \leq 180
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1,2,\ldots,10
$$

Where:
- $x_i$ = number of units of bread type $i$ to order each day (integer, $\geq 0$)
- $item\_value$ = expected profit per unit of bread type $i$
- $resource\_requirement$ = storage space required per unit of bread type $i$
- The total storage used by all bread types cannot exceed $180$ units.