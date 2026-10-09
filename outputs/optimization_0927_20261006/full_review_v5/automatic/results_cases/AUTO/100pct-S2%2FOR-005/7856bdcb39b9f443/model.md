#### Sets and Indices

Let $i$ index the bread types, with the following mapping (in source order):

| $i$ | item_name         |
|-----|-------------------|
| 1   | Baguette          |
| 2   | Croissant         |
| 3   | Sourdough         |
| 4   | Rye Bread         |
| 5   | Brioche           |
| 6   | Focaccia          |
| 7   | Ciabatta          |
| 8   | Pita              |
| 9   | Bagel             |
| 10  | English Muffin    |

#### Parameters

- $p_i$: expected profit per unit of bread type $i$ (item_value)
- $a_i$: storage space required per unit of bread type $i$ (resource_requirement)
- $C$: total storage capacity (resource_capacity)

Parameter values (from the data):

| item_name         | $p_i$ | $a_i$ |
|-------------------|-------|-------|
| Baguette          | 888   | 4     |
| Croissant         | 134   | 2     |
| Sourdough         | 129   | 4     |
| Rye Bread         | 370   | 3     |
| Brioche           | 921   | 2     |
| Focaccia          | 765   | 1     |
| Ciabatta          | 154   | 2     |
| Pita              | 837   | 1     |
| Bagel             | 584   | 3     |
| English Muffin    | 365   | 3     |

Total storage capacity: $C = 180$

#### Decision Variables

- $x_i$: number of units of bread type $i$ to order each day, $x_i \in \mathbb{Z}_{\geq 0}$

#### Mathematical Model

**Objective:**
\[
\max \sum_{i=1}^{10} p_i x_i
\]
That is,
\[
\max \Big(
888\,x_1 + 134\,x_2 + 129\,x_3 + 370\,x_4 + 921\,x_5 + 765\,x_6 + 154\,x_7 + 837\,x_8 + 584\,x_9 + 365\,x_{10}
\Big)
\]

**Subject to:**

Storage capacity constraint:
\[
4x_1 + 2x_2 + 4x_3 + 3x_4 + 2x_5 + 1x_6 + 2x_7 + 1x_8 + 3x_9 + 3x_{10} \leq 180
\]

Non-negativity and integrality:
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1,\ldots,10
\]

#### Parameter Table (source order)

| item_name         | item_value ($p_i$) | resource_requirement ($a_i$) |
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

Total storage capacity: $C = 180$

#### Variable Domains

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]