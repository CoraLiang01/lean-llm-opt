Let $x_i$ be the number of units of bread type $i$ to order each day. The index $i$ runs over the following bread types, in the order given:

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

Let $p_i$ be the expected profit per unit of bread $i$, and $a_i$ be the storage space required per unit of bread $i$. The total storage capacity is $180$.

The model is:

$$
\begin{align*}
\max\quad & 888\,x_1 + 134\,x_2 + 129\,x_3 + 370\,x_4 + 921\,x_5 + 765\,x_6 + 154\,x_7 + 837\,x_8 + 584\,x_9 + 365\,x_{10} \\[2ex]
\text{s.t.}\quad & 4\,x_1 + 2\,x_2 + 4\,x_3 + 3\,x_4 + 2\,x_5 + 1\,x_6 + 2\,x_7 + 1\,x_8 + 3\,x_9 + 3\,x_{10} \leq 180 \\[2ex]
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1,\ldots,10
\end{align*}
$$

Where:

- $x_1$: Baguette
- $x_2$: Croissant
- $x_3$: Sourdough
- $x_4$: Rye Bread
- $x_5$: Brioche
- $x_6$: Focaccia
- $x_7$: Ciabatta
- $x_8$: Pita
- $x_9$: Bagel
- $x_{10}$: English Muffin

All variables are nonnegative integers. The objective is to maximize total expected profit, subject to the daily storage capacity constraint.