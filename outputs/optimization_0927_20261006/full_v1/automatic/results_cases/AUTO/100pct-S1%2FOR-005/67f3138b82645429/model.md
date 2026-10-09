Let $x_i$ be the number of units of bread type $i$ to order each day. Each $x_i$ is a nonnegative integer.

Let the set of bread types (in source order) be:
- Baguette
- Croissant
- Sourdough
- Rye Bread
- Brioche
- Focaccia
- Ciabatta
- Pita
- Bagel
- English Muffin

Let $p_i$ be the expected profit per unit, and $a_i$ be the storage space required per unit, as follows:

| item_name        | $p_i$ (item_value) | $a_i$ (resource_requirement) |
|------------------|-------------------|-----------------------------|
| Baguette         | 888               | 4                           |
| Croissant        | 134               | 2                           |
| Sourdough        | 129               | 4                           |
| Rye Bread        | 370               | 3                           |
| Brioche          | 921               | 2                           |
| Focaccia         | 765               | 1                           |
| Ciabatta         | 154               | 2                           |
| Pita             | 837               | 1                           |
| Bagel            | 584               | 3                           |
| English Muffin   | 365               | 3                           |

The total storage capacity is $180$ units.

The mathematical model is:

Objective:
$$
\max \; 888x_{\text{Baguette}} + 134x_{\text{Croissant}} + 129x_{\text{Sourdough}} + 370x_{\text{Rye Bread}} + 921x_{\text{Brioche}} + 765x_{\text{Focaccia}} + 154x_{\text{Ciabatta}} + 837x_{\text{Pita}} + 584x_{\text{Bagel}} + 365x_{\text{English Muffin}}
$$

Subject to:
$$
4x_{\text{Baguette}} + 2x_{\text{Croissant}} + 4x_{\text{Sourdough}} + 3x_{\text{Rye Bread}} + 2x_{\text{Brioche}} + 1x_{\text{Focaccia}} + 2x_{\text{Ciabatta}} + 1x_{\text{Pita}} + 3x_{\text{Bagel}} + 3x_{\text{English Muffin}} \leq 180
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \text{for all bread types } i
$$

Where the bread types $i$ are as listed above, and all coefficients are as given in the retrieved data.