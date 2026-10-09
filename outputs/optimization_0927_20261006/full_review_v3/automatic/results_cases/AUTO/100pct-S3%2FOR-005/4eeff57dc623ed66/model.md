Let $x_i$ be the number of units of bread type $i$ to order each day, where $i$ indexes the bread types as listed in the original data.

#### Sets and Parameters

- Let $I$ be the set of bread types, with $i$ corresponding to the following (in source order):

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

- Let $v_i$ be the expected profit per unit of bread type $i$ ("item_value").
- Let $a_i$ be the storage requirement per unit of bread type $i$ ("resource_requirement").
- Let $C$ be the total storage capacity ("resource_capacity").

#### Data

From the retrieved data:

- Storage capacity: $C = 180$
- Bread types, profits, and storage requirements:

| $i$ | item_name         | $v_i$ | $a_i$ |
|-----|-------------------|-------|-------|
| 1   | Baguette          | 888   | 4     |
| 2   | Croissant         | 134   | 2     |
| 3   | Sourdough         | 129   | 4     |
| 4   | Rye Bread         | 370   | 3     |
| 5   | Brioche           | 921   | 2     |
| 6   | Focaccia          | 765   | 1     |
| 7   | Ciabatta          | 154   | 2     |
| 8   | Pita              | 837   | 1     |
| 9   | Bagel             | 584   | 3     |
| 10  | English Muffin    | 365   | 3     |

#### Decision Variables

$x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$ (number of units of bread type $i$ to order each day; must be integer and nonnegative).

#### Mathematical Model

**Objective:**
\[
\max \sum_{i=1}^{10} v_i x_i = 888x_1 + 134x_2 + 129x_3 + 370x_4 + 921x_5 + 765x_6 + 154x_7 + 837x_8 + 584x_9 + 365x_{10}
\]

**Subject to:**
\[
4x_1 + 2x_2 + 4x_3 + 3x_4 + 2x_5 + 1x_6 + 2x_7 + 1x_8 + 3x_9 + 3x_{10} \leq 180
\]

\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 10
\]

**Where:**

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

**All variables are nonnegative integers.**