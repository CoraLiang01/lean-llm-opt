##### Sets and Indices

Let $i$ index the bread types, corresponding to the item_name in products.csv.

##### Parameters

For each bread type $i$:
- $p_i$ = item_value of bread $i$ (expected profit per unit)
- $a_i$ = resource_requirement of bread $i$ (storage space per unit)

Let $C$ = resource_capacity from capacity.csv (total available storage space per day)

##### Decision Variables

For each bread type $i$:
- $x_i$ = number of units of bread $i$ to order each day ($x_i \in \mathbb{Z}_{\geq 0}$)

##### Mathematical Model

Objective:
\[
\max \sum_{i} p_i x_i
\]
where $p_i$ and $x_i$ are as defined above.

Subject to:
\[
\sum_{i} a_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]

##### Data

From products.csv (in source order):

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

From capacity.csv:

| resource_capacity |
|-------------------|
| 180               |

##### Complete Numerical Formulation

Let the bread types $i$ be, in order:
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

Let $x_i$ be the integer number of units of bread $i$ to order.

\[
\max \left(
888x_1 + 134x_2 + 129x_3 + 370x_4 + 921x_5 + 765x_6 + 154x_7 + 837x_8 + 584x_9 + 365x_{10}
\right)
\]

Subject to:

\[
4x_1 + 2x_2 + 4x_3 + 3x_4 + 2x_5 + 1x_6 + 2x_7 + 1x_8 + 3x_9 + 3x_{10} \leq 180
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1,\ldots,10
\]