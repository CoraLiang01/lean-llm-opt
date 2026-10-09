Let $x_i$ be the number of units of bread type $i$ to order each day. Each $x_i$ is a nonnegative integer.

Define the following sets and parameters, using the exact identifiers from the data:

- Let $I$ be the set of bread types (item_name):

  $I = \{$Baguette, Croissant, Sourdough, Rye Bread, Brioche, Focaccia, Ciabatta, Pita, Bagel, English Muffin$\}$

- For each $i \in I$:
    - $v_i$ = item_value (expected profit per unit)
    - $a_i$ = resource_requirement (storage space per unit)

- Let $C$ = resource_capacity (total available storage space per day)

Parameters from the data:

| item_name        | item_value | resource_requirement |
|------------------|-----------|---------------------|
| Baguette         | 888       | 4                   |
| Croissant        | 134       | 2                   |
| Sourdough        | 129       | 4                   |
| Rye Bread        | 370       | 3                   |
| Brioche          | 921       | 2                   |
| Focaccia         | 765       | 1                   |
| Ciabatta         | 154       | 2                   |
| Pita             | 837       | 1                   |
| Bagel            | 584       | 3                   |
| English Muffin   | 365       | 3                   |

Total storage capacity per day: $C = 180$

The mathematical model is:

$$
\begin{align*}
\max \quad & 888\,x_{\text{Baguette}} + 134\,x_{\text{Croissant}} + 129\,x_{\text{Sourdough}} + 370\,x_{\text{Rye Bread}} \\
& + 921\,x_{\text{Brioche}} + 765\,x_{\text{Focaccia}} + 154\,x_{\text{Ciabatta}} + 837\,x_{\text{Pita}} \\
& + 584\,x_{\text{Bagel}} + 365\,x_{\text{English Muffin}} \\
\text{s.t.} \quad & 4\,x_{\text{Baguette}} + 2\,x_{\text{Croissant}} + 4\,x_{\text{Sourdough}} + 3\,x_{\text{Rye Bread}} \\
& + 2\,x_{\text{Brioche}} + 1\,x_{\text{Focaccia}} + 2\,x_{\text{Ciabatta}} + 1\,x_{\text{Pita}} \\
& + 3\,x_{\text{Bagel}} + 3\,x_{\text{English Muffin}} \leq 180 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\end{align*}
$$

Where:
- $x_i$ is the number of units of bread type $i$ to order each day (integer, $\geq 0$)
- The objective maximizes total expected profit.
- The constraint ensures total storage used does not exceed 180 units.