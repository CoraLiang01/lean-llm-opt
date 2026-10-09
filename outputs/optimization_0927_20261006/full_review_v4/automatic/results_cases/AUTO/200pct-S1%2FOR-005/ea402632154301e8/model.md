Let $x_i$ be the number of units of bread type $i$ to order each day, where $i$ indexes the following bread types in the order retrieved:

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

Let the expected profit per unit and storage requirement per unit for each bread type be as follows:

| Bread Type        | item_value | resource_requirement |
|-------------------|------------|---------------------|
| Baguette          | 888        | 4                   |
| Croissant         | 134        | 2                   |
| Sourdough         | 129        | 4                   |
| Rye Bread         | 370        | 3                   |
| Brioche           | 921        | 2                   |
| Focaccia          | 765        | 1                   |
| Ciabatta          | 154        | 2                   |
| Pita              | 837        | 1                   |
| Bagel             | 584        | 3                   |
| English Muffin    | 365        | 3                   |

The total storage capacity is 180 units.

The mathematical model is:

$$
\begin{align*}
\max \quad & 888\,x_1 + 134\,x_2 + 129\,x_3 + 370\,x_4 + 921\,x_5 + 765\,x_6 + 154\,x_7 + 837\,x_8 + 584\,x_9 + 365\,x_{10} \\
\text{s.t.} \quad & 4\,x_1 + 2\,x_2 + 4\,x_3 + 3\,x_4 + 2\,x_5 + 1\,x_6 + 2\,x_7 + 1\,x_8 + 3\,x_9 + 3\,x_{10} \leq 180 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1,\ldots,10
\end{align*}
$$

Where:
- $x_i$ is the number of units of bread type $i$ to order each day (nonnegative integer).
- The objective maximizes total expected profit.
- The constraint ensures total storage used does not exceed 180 units.