Let $x_i$ be the number of units of bread type $i$ to order each day, where $i$ indexes the bread types as given by the item_name column in products.csv. All $x_i$ are nonnegative integers.

Let $p_i$ be the expected profit per unit of bread type $i$ (from item_value in products.csv).

Let $a_i$ be the resource requirement per unit of bread type $i$ (from resource_requirement in products.csv).

Let $C$ be the total storage capacity (resource_capacity from capacity.csv).

Define:

- Bread types (in source order):

  1. Baguette: $p_1 = 888$, $a_1 = 4$
  2. Croissant: $p_2 = 134$, $a_2 = 2$
  3. Sourdough: $p_3 = 129$, $a_3 = 4$
  4. Rye Bread: $p_4 = 370$, $a_4 = 3$
  5. Brioche: $p_5 = 921$, $a_5 = 2$
  6. Focaccia: $p_6 = 765$, $a_6 = 1$
  7. Ciabatta: $p_7 = 154$, $a_7 = 2$
  8. Pita: $p_8 = 837$, $a_8 = 1$
  9. Bagel: $p_9 = 584$, $a_9 = 3$
  10. English Muffin: $p_{10} = 365$, $a_{10} = 3$

- Storage capacity: $C = 180$

The mathematical model is:

$$
\begin{align*}
\max \quad & 888x_1 + 134x_2 + 129x_3 + 370x_4 + 921x_5 + 765x_6 + 154x_7 + 837x_8 + 584x_9 + 365x_{10} \\
\text{s.t.} \quad & 4x_1 + 2x_2 + 4x_3 + 3x_4 + 2x_5 + 1x_6 + 2x_7 + 1x_8 + 3x_9 + 3x_{10} \leq 180 \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1,\ldots,10
\end{align*}
$$

Where:
- $x_1$ = Baguette
- $x_2$ = Croissant
- $x_3$ = Sourdough
- $x_4$ = Rye Bread
- $x_5$ = Brioche
- $x_6$ = Focaccia
- $x_7$ = Ciabatta
- $x_8$ = Pita
- $x_9$ = Bagel
- $x_{10}$ = English Muffin

All coefficients and identifiers are as retrieved from the data.