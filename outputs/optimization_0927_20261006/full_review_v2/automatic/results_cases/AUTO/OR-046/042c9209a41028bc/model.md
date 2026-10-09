Let $x_i$ be the number of units of product $i$ to order each day, where $i$ indexes the products in the order given below.

**Products (in source order):**

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

**Parameters:**

- $v_i$: Value per unit of product $i$
- $w_i$: Weight per unit of product $i$
- $C$: Total stock capacity

**Data:**

| $i$ | Product Name           | $w_i$ (Weight) | $v_i$ (Value) |
|-----|------------------------|----------------|---------------|
| 1   | Spinach                | 230            | 64            |
| 2   | Shiitake Mushrooms     | 637            | 75            |
| 3   | Apples                 | 773            | 68            |
| 4   | Carrots                | 653            | 11            |
| 5   | Basil                  | 755            | 91            |
| 6   | Potatoes               | 670            | 31            |
| 7   | Green Beans            | 505            | 90            |
| 8   | Blueberries            | 821            | 56            |
| 9   | Oranges                | 83             | 10            |
| 10  | Watermelons            | 249            | 24            |

Total stock capacity: $C = 875$

---

### Mathematical Model

**Objective:**

$$
\max \; 64x_1 + 75x_2 + 68x_3 + 11x_4 + 91x_5 + 31x_6 + 90x_7 + 56x_8 + 10x_9 + 24x_{10}
$$

**Subject to:**

$$
230x_1 + 637x_2 + 773x_3 + 653x_4 + 755x_5 + 670x_6 + 505x_7 + 821x_8 + 83x_9 + 249x_{10} \leq 875
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 10
$$

Where:

- $x_i$ = number of units of product $i$ to order each day (nonnegative integer)
- All coefficients and product names are as listed above, in the original source order.