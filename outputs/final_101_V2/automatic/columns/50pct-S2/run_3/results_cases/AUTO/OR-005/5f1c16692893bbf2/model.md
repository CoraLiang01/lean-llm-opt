Let $x_i$ be the number of units of bread type $i$ to order each day, where $i$ indexes the rows of products.csv in the given order.

Maximize total expected profit:
$$
\max \left( 888x_1 + 134x_2 + 129x_3 + 370x_4 + 921x_5 + 765x_6 + 154x_7 + 837x_8 + 584x_9 + 365x_{10} \right)
$$

Subject to the storage capacity constraint:
$$
4x_1 + 2x_2 + 4x_3 + 3x_4 + 2x_5 + 1x_6 + 2x_7 + 1x_8 + 3x_9 + 3x_{10} \leq 180
$$

Variable domains:
$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 10
$$

Where the mapping of $i$ to bread type is:
\begin{align*}
x_1 &: \text{Baguette} \\
x_2 &: \text{Croissant} \\
x_3 &: \text{Sourdough} \\
x_4 &: \text{Rye Bread} \\
x_5 &: \text{Brioche} \\
x_6 &: \text{Focaccia} \\
x_7 &: \text{Ciabatta} \\
x_8 &: \text{Pita} \\
x_9 &: \text{Bagel} \\
x_{10} &: \text{English Muffin}
\end{align*}

All coefficients and identifiers are as retrieved from the source files.