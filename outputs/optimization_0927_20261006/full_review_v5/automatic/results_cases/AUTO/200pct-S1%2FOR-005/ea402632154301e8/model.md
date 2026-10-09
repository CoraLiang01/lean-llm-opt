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

Let $p_i$ be the expected profit per unit of bread type $i$, and $a_i$ be the storage space required per unit of bread type $i$. The total available storage capacity is $C$.

#### Parameter Table (in source order):

| item_name        | item_value | resource_requirement |
|------------------|------------|---------------------|
| Baguette         | 888        | 4                   |
| Croissant        | 134        | 2                   |
| Sourdough        | 129        | 4                   |
| Rye Bread        | 370        | 3                   |
| Brioche          | 921        | 2                   |
| Focaccia         | 765        | 1                   |
| Ciabatta         | 154        | 2                   |
| Pita             | 837        | 1                   |
| Bagel            | 584        | 3                   |
| English Muffin   | 365        | 3                   |

Total storage capacity: $C = 180$

#### Mathematical Model

**Objective:**
\[
\max \; 888x_1 + 134x_2 + 129x_3 + 370x_4 + 921x_5 + 765x_6 + 154x_7 + 837x_8 + 584x_9 + 365x_{10}
\]

**Subject to:**
\[
4x_1 + 2x_2 + 4x_3 + 3x_4 + 2x_5 + 1x_6 + 2x_7 + 1x_8 + 3x_9 + 3x_{10} \leq 180
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 10
\]

**Where:**

- $x_1$ = units of Baguette
- $x_2$ = units of Croissant
- $x_3$ = units of Sourdough
- $x_4$ = units of Rye Bread
- $x_5$ = units of Brioche
- $x_6$ = units of Focaccia
- $x_7$ = units of Ciabatta
- $x_8$ = units of Pita
- $x_9$ = units of Bagel
- $x_{10}$ = units of English Muffin

All variables are nonnegative integers.