Let $x_i$ be the number of units of product $i$ to order each day, where $i$ indexes the following products in the order given:

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

Let $v_i$ be the value (profit) per unit and $w_i$ the weight (space requirement) per unit for product $i$ as given below:

| $i$ | ProductName           | $w_i$ (Weight) | $v_i$ (Value) |
|-----|----------------------|----------------|---------------|
| 1   | Spinach              | 230            | 64            |
| 2   | Shiitake Mushrooms   | 637            | 75            |
| 3   | Apples               | 773            | 68            |
| 4   | Carrots              | 653            | 11            |
| 5   | Basil                | 755            | 91            |
| 6   | Potatoes             | 670            | 31            |
| 7   | Green Beans          | 505            | 90            |
| 8   | Blueberries          | 821            | 56            |
| 9   | Oranges              | 83             | 10            |
| 10  | Watermelons          | 249            | 24            |

The total stock capacity is $C = 875$.

The mathematical model is:

Maximize total value:
$$
\max \; 64x_1 + 75x_2 + 68x_3 + 11x_4 + 91x_5 + 31x_6 + 90x_7 + 56x_8 + 10x_9 + 24x_{10}
$$

Subject to the stock capacity constraint:
$$
230x_1 + 637x_2 + 773x_3 + 653x_4 + 755x_5 + 670x_6 + 505x_7 + 821x_8 + 83x_9 + 249x_{10} \leq 875
$$

Nonnegativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 10
$$

Where:
- $x_i$ = number of units of product $i$ to order each day (integer, $\geq 0$)
- $w_i$ = weight (space requirement) per unit of product $i$
- $v_i$ = value (profit) per unit of product $i$
- $C = 875$ is the total stock capacity.