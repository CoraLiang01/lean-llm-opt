Let $x_i$ be the number of units of bread type $i$ to order each day. Each $x_i$ is a nonnegative integer.

Define the following parameters for each bread type $i$ (using the original identifiers):

- $v_i$: expected profit per unit (item_value)
- $a_i$: storage space required per unit (resource_requirement)

The total available storage capacity is $180$ units (resource_capacity).

The bread types and their parameters are:

| item_name         | item_value ($v_i$) | resource_requirement ($a_i$) |
|-------------------|-------------------|------------------------------|
| Baguette          | 888               | 4                            |
| Croissant         | 134               | 2                            |
| Sourdough         | 129               | 4                            |
| Rye Bread         | 370               | 3                            |
| Brioche           | 921               | 2                            |
| Focaccia          | 765               | 1                            |
| Ciabatta          | 154               | 2                            |
| Pita              | 837               | 1                            |
| Bagel             | 584               | 3                            |
| English Muffin    | 365               | 3                            |

The mathematical model is:

$$
\begin{align*}
\text{Maximize} \quad & 888\,x_{\text{Baguette}} + 134\,x_{\text{Croissant}} + 129\,x_{\text{Sourdough}} + 370\,x_{\text{Rye Bread}} \\
& + 921\,x_{\text{Brioche}} + 765\,x_{\text{Focaccia}} + 154\,x_{\text{Ciabatta}} + 837\,x_{\text{Pita}} \\
& + 584\,x_{\text{Bagel}} + 365\,x_{\text{English Muffin}} \\
\text{subject to} \quad & 4\,x_{\text{Baguette}} + 2\,x_{\text{Croissant}} + 4\,x_{\text{Sourdough}} + 3\,x_{\text{Rye Bread}} \\
& + 2\,x_{\text{Brioche}} + 1\,x_{\text{Focaccia}} + 2\,x_{\text{Ciabatta}} + 1\,x_{\text{Pita}} \\
& + 3\,x_{\text{Bagel}} + 3\,x_{\text{English Muffin}} \leq 180 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \text{for all bread types } i
\end{align*}
$$

Where the variables $x_{\text{Baguette}}, x_{\text{Croissant}}, x_{\text{Sourdough}}, x_{\text{Rye Bread}}, x_{\text{Brioche}}, x_{\text{Focaccia}}, x_{\text{Ciabatta}}, x_{\text{Pita}}, x_{\text{Bagel}}, x_{\text{English Muffin}}$ represent the number of units of each bread type to order each day.