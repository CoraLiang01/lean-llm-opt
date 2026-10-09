Let $x_i$ be the number of units of bread type $i$ to order each day, where $i$ indexes the bread types as listed in the data.

**Parameters:**

- Let $p_i$ be the expected profit per unit of bread type $i$ (from Value).
- Let $w_i$ be the storage space required per unit of bread type $i$ (from Weight).
- Let $C$ be the total available storage capacity (from Capacity).

**Data:**

| $i$ | ProductName       | $p_i$ (Value) | $w_i$ (Weight) |
|-----|------------------|---------------|---------------|
| 1   | Baguette         | 888           | 4             |
| 2   | Croissant        | 134           | 2             |
| 3   | Sourdough        | 129           | 4             |
| 4   | Rye Bread        | 370           | 3             |
| 5   | Brioche          | 921           | 2             |
| 6   | Focaccia         | 765           | 1             |
| 7   | Ciabatta         | 154           | 2             |
| 8   | Pita             | 837           | 1             |
| 9   | Bagel            | 584           | 3             |
| 10  | English Muffin   | 365           | 3             |

Total storage capacity: $C = 180$

---

**Mathematical Model:**

Maximize total expected profit:
$$
\max \; 888x_1 + 134x_2 + 129x_3 + 370x_4 + 921x_5 + 765x_6 + 154x_7 + 837x_8 + 584x_9 + 365x_{10}
$$

Subject to the storage capacity constraint:
$$
4x_1 + 2x_2 + 4x_3 + 3x_4 + 2x_5 + 1x_6 + 2x_7 + 1x_8 + 3x_9 + 3x_{10} \leq 180
$$

And integrality and non-negativity:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 10
$$

Where:

- $x_1$ = units of Baguette to order
- $x_2$ = units of Croissant to order
- $x_3$ = units of Sourdough to order
- $x_4$ = units of Rye Bread to order
- $x_5$ = units of Brioche to order
- $x_6$ = units of Focaccia to order
- $x_7$ = units of Ciabatta to order
- $x_8$ = units of Pita to order
- $x_9$ = units of Bagel to order
- $x_{10}$ = units of English Muffin to order