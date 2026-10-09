Let $x_i$ be the number of units of bread type $i$ to order each day. Each $x_i$ is a nonnegative integer.

Define the following parameters for each bread type $i$ (using the item_name as $i$):

- $p_i$: expected profit per unit of bread $i$ (item_value)
- $a_i$: storage space required per unit of bread $i$ (resource_requirement)

Let $C$ be the total storage capacity (resource_capacity).

The data is:

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

Total storage capacity: $C = 180$

The mathematical model is:

$$
\begin{align*}
\max \quad & 888x_{\text{Baguette}} + 134x_{\text{Croissant}} + 129x_{\text{Sourdough}} + 370x_{\text{Rye Bread}} + 921x_{\text{Brioche}} \\
& + 765x_{\text{Focaccia}} + 154x_{\text{Ciabatta}} + 837x_{\text{Pita}} + 584x_{\text{Bagel}} + 365x_{\text{English Muffin}} \\[2ex]
\text{s.t.} \quad & 4x_{\text{Baguette}} + 2x_{\text{Croissant}} + 4x_{\text{Sourdough}} + 3x_{\text{Rye Bread}} + 2x_{\text{Brioche}} \\
& + 1x_{\text{Focaccia}} + 2x_{\text{Ciabatta}} + 1x_{\text{Pita}} + 3x_{\text{Bagel}} + 3x_{\text{English Muffin}} \leq 180 \\[2ex]
& x_i \in \mathbb{Z}_{\geq 0} \quad \text{for all bread types } i
\end{align*}
$$

Where the bread types $i$ are:
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