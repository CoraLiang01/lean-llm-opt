Let $x_i$ be the number of units of product $i$ to order each day, for each product $i$ in the list below.

**Products and Parameters:**

| $i$ | Product Name           | Weight ($w_i$) | Value ($v_i$) |
|-----|------------------------|---------------|--------------|
| 1   | Spinach                | 230           | 64           |
| 2   | Shiitake Mushrooms     | 637           | 75           |
| 3   | Apples                 | 773           | 68           |
| 4   | Carrots                | 653           | 11           |
| 5   | Basil                  | 755           | 91           |
| 6   | Potatoes               | 670           | 31           |
| 7   | Green Beans            | 505           | 90           |
| 8   | Blueberries            | 821           | 56           |
| 9   | Oranges                | 83            | 10           |
| 10  | Watermelons            | 249           | 24           |

Total stock capacity: $C = 875$

**Mathematical Model:**

Maximize total value:
$$
\max \; 64x_1 + 75x_2 + 68x_3 + 11x_4 + 91x_5 + 31x_6 + 90x_7 + 56x_8 + 10x_9 + 24x_{10}
$$

Subject to the overall capacity constraint:
$$
230x_1 + 637x_2 + 773x_3 + 653x_4 + 755x_5 + 670x_6 + 505x_7 + 821x_8 + 83x_9 + 249x_{10} \leq 875
$$

And integrality/nonnegativity:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 10
$$

Where:
- $x_1$ = Spinach
- $x_2$ = Shiitake Mushrooms
- $x_3$ = Apples
- $x_4$ = Carrots
- $x_5$ = Basil
- $x_6$ = Potatoes
- $x_7$ = Green Beans
- $x_8$ = Blueberries
- $x_9$ = Oranges
- $x_{10}$ = Watermelons