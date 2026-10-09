Let $x_i$ be the number of units of bread type $i$ to order each day, where $i$ indexes the bread types as listed below. All $x_i$ are nonnegative integers.

**Parameters:**

- $p_i$: expected profit per unit of bread type $i$ ("item_value")
- $a_i$: storage space required per unit of bread type $i$ ("resource_requirement")
- $C$: total available storage capacity ("resource_capacity" = 180)

**Bread Types, Profits, and Space Requirements:**

| $i$ | item_name        | $p_i$ (item_value) | $a_i$ (resource_requirement) |
|-----|------------------|-------------------|------------------------------|
| 1   | Baguette         | 888               | 4                            |
| 2   | Croissant        | 134               | 2                            |
| 3   | Sourdough        | 129               | 4                            |
| 4   | Rye Bread        | 370               | 3                            |
| 5   | Focaccia         | 765               | 1                            |
| 6   | Ciabatta         | 154               | 2                            |
| 7   | Pita             | 837               | 1                            |
| 8   | Bagel            | 584               | 3                            |
| 9   | English Muffin   | 365               | 3                            |

**Storage Capacity:**

$C = 180$

---

### Mathematical Model

**Objective:**

$$
\max \; 888x_1 + 134x_2 + 129x_3 + 370x_4 + 765x_5 + 154x_6 + 837x_7 + 584x_8 + 365x_9
$$

**Subject to:**

$$
4x_1 + 2x_2 + 4x_3 + 3x_4 + 1x_5 + 2x_6 + 1x_7 + 3x_8 + 3x_9 \leq 180
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 9
$$

**Where:**

- $x_1$: Baguette
- $x_2$: Croissant
- $x_3$: Sourdough
- $x_4$: Rye Bread
- $x_5$: Focaccia
- $x_6$: Ciabatta
- $x_7$: Pita
- $x_8$: Bagel
- $x_9$: English Muffin

All variables are nonnegative integers.