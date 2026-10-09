Let $x_i$ be the number of units of product $i$ to order each day, where $i$ indexes the products in the order given below. All $x_i$ are nonnegative integers.

**Objective:**
\[
\max \; 64x_1 + 75x_2 + 68x_3 + 11x_4 + 91x_5 + 31x_6 + 90x_7 + 56x_8 + 10x_9 + 24x_{10}
\]

**Subject to:**

\[
230x_1 + 637x_2 + 773x_3 + 653x_4 + 755x_5 + 670x_6 + 505x_7 + 821x_8 + 83x_9 + 249x_{10} \leq 875
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 10
\]

**Where:**

| $i$ | ProductName           | Value | Weight |
|-----|-----------------------|-------|--------|
| 1   | Spinach               | 64    | 230    |
| 2   | Shiitake Mushrooms    | 75    | 637    |
| 3   | Apples                | 68    | 773    |
| 4   | Carrots               | 11    | 653    |
| 5   | Basil                 | 91    | 755    |
| 6   | Potatoes              | 31    | 670    |
| 7   | Green Beans           | 90    | 505    |
| 8   | Blueberries           | 56    | 821    |
| 9   | Oranges               | 10    | 83     |
| 10  | Watermelons           | 24    | 249    |

- The objective maximizes total benefit from all products ordered.
- The constraint ensures the total weight of all products ordered does not exceed the overall stock capacity of 875 units.
- All $x_i$ are nonnegative integers.