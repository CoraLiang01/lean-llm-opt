##### Sets and Indices

Let $i$ index the products as listed in the original order of products.csv:
1. Spinach
2. Shiitake Mushrooms
3. Apples
4. Carrots
5. Basil
6. Potatoes
7. Green Beans
8. Blueberries
9. Oranges
10. Watermelons

##### Parameters

- $v_i$: Value per unit of product $i$ (from Value column)
- $w_i$: Weight per unit of product $i$ (from Weight column)
- $C$: Total stock capacity (from Capacity column in capacity.csv)

##### Decision Variables

- $x_i$: Number of units of product $i$ to order each day ($x_i \in \mathbb{Z}_{\geq 0}$)

##### Data

From products.csv (in source order):

| $i$ | ProductName           | $w_i$ | $v_i$ |
|-----|-----------------------|-------|-------|
| 1   | Spinach               | 230   | 64    |
| 2   | Shiitake Mushrooms    | 637   | 75    |
| 3   | Apples                | 773   | 68    |
| 4   | Carrots               | 653   | 11    |
| 5   | Basil                 | 755   | 91    |
| 6   | Potatoes              | 670   | 31    |
| 7   | Green Beans           | 505   | 90    |
| 8   | Blueberries           | 821   | 56    |
| 9   | Oranges               | 83    | 10    |
| 10  | Watermelons           | 249   | 24    |

From capacity.csv:

$C = 875$

##### Mathematical Model

Objective:
\[
\max \; 64x_1 + 75x_2 + 68x_3 + 11x_4 + 91x_5 + 31x_6 + 90x_7 + 56x_8 + 10x_9 + 24x_{10}
\]

Subject to:
\[
230x_1 + 637x_2 + 773x_3 + 653x_4 + 755x_5 + 670x_6 + 505x_7 + 821x_8 + 83x_9 + 249x_{10} \leq 875
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 10
\]

##### Variable Definitions

- $x_i$: Number of units of product $i$ to order each day (nonnegative integer)

##### Parameter Definitions

- $v_i$: Value per unit of product $i$ (see table above)
- $w_i$: Weight per unit of product $i$ (see table above)
- $C$: Total stock capacity $= 875$

##### Complete Data Used

From capacity.csv:
- Capacity: 875

From products.csv (source order):
1. Spinach, Weight: 230, Value: 64
2. Shiitake Mushrooms, Weight: 637, Value: 75
3. Apples, Weight: 773, Value: 68
4. Carrots, Weight: 653, Value: 11
5. Basil, Weight: 755, Value: 91
6. Potatoes, Weight: 670, Value: 31
7. Green Beans, Weight: 505, Value: 90
8. Blueberries, Weight: 821, Value: 56
9. Oranges, Weight: 83, Value: 10
10. Watermelons, Weight: 249, Value: 24