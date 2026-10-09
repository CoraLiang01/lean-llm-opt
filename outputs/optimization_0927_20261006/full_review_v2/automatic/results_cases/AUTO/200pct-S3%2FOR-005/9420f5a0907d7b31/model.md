Let $x_i$ be the number of units of bread type $i$ to order each day. The index $i$ runs over the following bread types, with their associated expected profit $p_i$ and storage requirement $a_i$:

| item_name        | $p_i$ (item_value) | $a_i$ (resource_requirement) |
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

The total available storage capacity per day is $180$ units.

The mathematical model is:

$$
\begin{align*}
\text{Maximize} \quad & 888x_{\text{Baguette}} + 134x_{\text{Croissant}} + 129x_{\text{Sourdough}} + 370x_{\text{Rye Bread}} + 921x_{\text{Brioche}} \\
& + 765x_{\text{Focaccia}} + 154x_{\text{Ciabatta}} + 837x_{\text{Pita}} + 584x_{\text{Bagel}} + 365x_{\text{English Muffin}} \\[2ex]
\text{subject to} \quad & 4x_{\text{Baguette}} + 2x_{\text{Croissant}} + 4x_{\text{Sourdough}} + 3x_{\text{Rye Bread}} + 2x_{\text{Brioche}} \\
& + 1x_{\text{Focaccia}} + 2x_{\text{Ciabatta}} + 1x_{\text{Pita}} + 3x_{\text{Bagel}} + 3x_{\text{English Muffin}} \leq 180 \\[2ex]
& x_i \in \mathbb{Z}_{\geq 0} \quad \text{for all bread types } i
\end{align*}
$$

Where:
- $x_i$ = number of units of bread type $i$ to order each day (integer, $\geq 0$)
- $p_i$ = expected profit per unit of bread type $i$ (item_value)
- $a_i$ = storage requirement per unit of bread type $i$ (resource_requirement)
- Total storage capacity per day = $180$ units

All coefficients and identifiers are as retrieved and in source order.