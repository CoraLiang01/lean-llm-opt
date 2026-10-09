Let $x_i$ be the number of units of bread type $i$ to order each day, where $i$ indexes the bread types as listed in the data.

**Parameters:**

- $v_i$: expected profit per unit of bread type $i$ ("item_value")
- $a_i$: storage requirement per unit of bread type $i$ ("resource_requirement")
- $C$: total storage capacity ("resource_capacity")

**Data:**

- $C = 180$

| $i$ | item_name        | $v_i$ (item_value) | $a_i$ (resource_requirement) |
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

**Mathematical Model:**

Objective:
$$
\max \; 888x_1 + 134x_2 + 129x_3 + 370x_4 + 921x_5 + 765x_6 + 154x_7 + 837x_8 + 584x_9 + 365x_{10}
$$

Subject to:
$$
4x_1 + 2x_2 + 4x_3 + 3x_4 + 2x_5 + 1x_6 + 2x_7 + 1x_8 + 3x_9 + 3x_{10} \leq 180
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 10
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