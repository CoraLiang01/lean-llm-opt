Let $x_i$ be the number of units of bread type $i$ to order each day. The index $i$ runs over all bread types listed below, using the item_name as the identifier.

**Parameters:**

- For each bread type $i$:
    - $p_i$ = item_value (expected profit per unit)
    - $a_i$ = resource_requirement (storage space required per unit)
- Total storage capacity: $C = 180$

**Bread Types and Data (in source order):**

| item_name        | item_value ($p_i$) | resource_requirement ($a_i$) |
|------------------|-------------------|------------------------------|
| Baguette         | 888               | 4                            |
| Croissant        | 134               | 2                            |
| Sourdough        | 129               | 4                            |
| Rye Bread        | 370               | 3                            |
| Brioche          | 921               | 2                            |
| Focaccia         | 765               | 1                            |
| Ciabatta         | 154               | 2                            |
| Pita             | 837               | 1                            |
| Bagel            | 584               | 3                            |
| English Muffin   | 365               | 3                            |

**Mathematical Model:**

Objective:
$$
\max \left(
888\,x_{\text{Baguette}} +
134\,x_{\text{Croissant}} +
129\,x_{\text{Sourdough}} +
370\,x_{\text{Rye Bread}} +
921\,x_{\text{Brioche}} +
765\,x_{\text{Focaccia}} +
154\,x_{\text{Ciabatta}} +
837\,x_{\text{Pita}} +
584\,x_{\text{Bagel}} +
365\,x_{\text{English Muffin}}
\right)
$$

Subject to:
$$
4\,x_{\text{Baguette}} +
2\,x_{\text{Croissant}} +
4\,x_{\text{Sourdough}} +
3\,x_{\text{Rye Bread}} +
2\,x_{\text{Brioche}} +
1\,x_{\text{Focaccia}} +
2\,x_{\text{Ciabatta}} +
1\,x_{\text{Pita}} +
3\,x_{\text{Bagel}} +
3\,x_{\text{English Muffin}}
\leq 180
$$

$$
x_{\text{Baguette}},\ x_{\text{Croissant}},\ x_{\text{Sourdough}},\ x_{\text{Rye Bread}},\ x_{\text{Brioche}},\ x_{\text{Focaccia}},\ x_{\text{Ciabatta}},\ x_{\text{Pita}},\ x_{\text{Bagel}},\ x_{\text{English Muffin}} \in \mathbb{Z}_{\geq 0}
$$

Where each $x_i$ is the number of units of bread type $i$ to order each day.