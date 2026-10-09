Let $x_i$ be the number of units of product $i$ to order each day, for each product $i$ in the list below.

**Products, Weights, and Values:**

| $i$ | ProductName           | Weight | Value |
|-----|-----------------------|--------|-------|
| 1   | Spinach               | 230    | 64    |
| 2   | Shiitake Mushrooms    | 637    | 75    |
| 3   | Apples                | 773    | 68    |
| 4   | Carrots               | 653    | 11    |
| 5   | Basil                 | 755    | 91    |
| 6   | Potatoes              | 670    | 31    |
| 7   | Green Beans           | 505    | 90    |
| 8   | Blueberries           | 821    | 56    |
| 9   | Oranges               | 83     | 10    |
| 10  | Watermelons           | 249    | 24    |

**Capacity:**
- Total stock capacity: $875$

**Mathematical Model:**

Objective:
$$
\max \; 64x_1 + 75x_2 + 68x_3 + 11x_4 + 91x_5 + 31x_6 + 90x_7 + 56x_8 + 10x_9 + 24x_{10}
$$

Subject to:
$$
230x_1 + 637x_2 + 773x_3 + 653x_4 + 755x_5 + 670x_6 + 505x_7 + 821x_8 + 83x_9 + 249x_{10} \leq 875
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \text{for } i = 1,2,\ldots,10
$$

Where:
- $x_1$ = units of Spinach to order
- $x_2$ = units of Shiitake Mushrooms to order
- $x_3$ = units of Apples to order
- $x_4$ = units of Carrots to order
- $x_5$ = units of Basil to order
- $x_6$ = units of Potatoes to order
- $x_7$ = units of Green Beans to order
- $x_8$ = units of Blueberries to order
- $x_9$ = units of Oranges to order
- $x_{10}$ = units of Watermelons to order