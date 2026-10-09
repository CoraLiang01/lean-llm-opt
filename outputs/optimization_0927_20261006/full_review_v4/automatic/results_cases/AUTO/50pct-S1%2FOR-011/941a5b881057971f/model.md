Let $x_i$ be the number of units of product $i$ to order each day, where $i$ indexes the products as listed below.

##### Sets and Parameters

- Products (in source order):

  1. Spinach: Value = 64, Weight = 230
  2. Shiitake Mushrooms: Value = 75, Weight = 637
  3. Apples: Value = 68, Weight = 773
  4. Carrots: Value = 11, Weight = 653
  5. Basil: Value = 91, Weight = 755
  6. Potatoes: Value = 31, Weight = 670
  7. Green Beans: Value = 90, Weight = 505
  8. Blueberries: Value = 56, Weight = 821
  9. Oranges: Value = 10, Weight = 83
  10. Watermelons: Value = 24, Weight = 249

- Total stock capacity: $875$

##### Decision Variables

- $x_i \in \mathbb{Z}_{\geq 0}$, for $i = 1, \ldots, 10$

##### Objective Function

\[
\max \; 64x_1 + 75x_2 + 68x_3 + 11x_4 + 91x_5 + 31x_6 + 90x_7 + 56x_8 + 10x_9 + 24x_{10}
\]

##### Constraints

\[
230x_1 + 637x_2 + 773x_3 + 653x_4 + 755x_5 + 670x_6 + 505x_7 + 821x_8 + 83x_9 + 249x_{10} \leq 875
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 10
\]

##### Product Index Reference

| $i$ | Product Name           | Value | Weight |
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