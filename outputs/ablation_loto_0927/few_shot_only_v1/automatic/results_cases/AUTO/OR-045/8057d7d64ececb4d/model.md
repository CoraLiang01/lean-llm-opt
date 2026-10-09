**Mathematical Model**

Let $x_i$ be the number of units of produce type $i$ to order daily, where $i$ indexes the following products in the order given:

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

Let $v_i$ be the value (benefit) per unit of product $i$, and $w_i$ be the weight per unit of product $i$, as given below:

| $i$ | ProductName           | $w_i$ (Weight) | $v_i$ (Value) |
|-----|-----------------------|:--------------:|:-------------:|
| 1   | Spinach               | 282            | 49            |
| 2   | Shiitake Mushrooms    | 83             | 30            |
| 3   | Apples                | 251            | 30            |
| 4   | Carrots               | 257            | 18            |
| 5   | Basil                 | 88             | 54            |
| 6   | Potatoes              | 52             | 27            |
| 7   | Green Beans           | 198            | 91            |
| 8   | Blueberries           | 203            | 88            |
| 9   | Oranges               | 87             | 78            |
| 10  | Watermelons           | 265            | 22            |

The total inventory capacity is $1035$ (from capacity.csv).

---

**Objective:**

$$
\max \left(49x_1 + 30x_2 + 30x_3 + 18x_4 + 54x_5 + 27x_6 + 91x_7 + 88x_8 + 78x_9 + 22x_{10}\right)
$$

**Subject to:**

$$
282x_1 + 83x_2 + 251x_3 + 257x_4 + 88x_5 + 52x_6 + 198x_7 + 203x_8 + 87x_9 + 265x_{10} \leq 1035
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 10
$$

---

**Where:**

- $x_i$ = number of units of product $i$ to order daily (integer, $\geq 0$)
- $v_i$ = value per unit of product $i$ (see table)
- $w_i$ = weight per unit of product $i$ (see table)
- Total weight of all ordered units cannot exceed $1035$ units.