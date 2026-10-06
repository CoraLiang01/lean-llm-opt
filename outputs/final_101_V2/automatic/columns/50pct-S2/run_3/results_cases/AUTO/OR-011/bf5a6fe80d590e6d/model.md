##### Sets and Indices

Let $i$ index the products in the order given in products.csv:
- 1: Spinach
- 2: Shiitake Mushrooms
- 3: Apples
- 4: Carrots
- 5: Basil
- 6: Potatoes
- 7: Green Beans
- 8: Blueberries
- 9: Oranges
- 10: Watermelons

##### Parameters

For each product $i$:
- $v_i$ = Value of product $i$ (from products.csv)
- $w_i$ = Weight of product $i$ (from products.csv)

Total stock capacity:
- $C = 875$ (from capacity.csv)

##### Decision Variables

For each product $i$:
- $x_i$ = number of units of product $i$ to order each day ($x_i \in \mathbb{Z}_{\geq 0}$)

##### Mathematical Model

Objective:
\[
\max \sum_{i=1}^{10} v_i x_i
\]

Subject to:
\[
\sum_{i=1}^{10} w_i x_i \leq 875
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 10
\]

##### Data

Product data (in source order):

| $i$ | ProductName           | $w_i$ (Weight) | $v_i$ (Value) |
|-----|-----------------------|----------------|---------------|
| 1   | Spinach               | 230            | 64            |
| 2   | Shiitake Mushrooms    | 637            | 75            |
| 3   | Apples                | 773            | 68            |
| 4   | Carrots               | 653            | 11            |
| 5   | Basil                 | 755            | 91            |
| 6   | Potatoes              | 670            | 31            |
| 7   | Green Beans           | 505            | 90            |
| 8   | Blueberries           | 821            | 56            |
| 9   | Oranges               | 83             | 10            |
| 10  | Watermelons           | 249            | 24            |

Capacity:
- $C = 875$

##### Complete Model (with coefficients):

\[
\max \ 64x_1 + 75x_2 + 68x_3 + 11x_4 + 91x_5 + 31x_6 + 90x_7 + 56x_8 + 10x_9 + 24x_{10}
\]
\[
230x_1 + 637x_2 + 773x_3 + 653x_4 + 755x_5 + 670x_6 + 505x_7 + 821x_8 + 83x_9 + 249x_{10} \leq 875
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 10
\]