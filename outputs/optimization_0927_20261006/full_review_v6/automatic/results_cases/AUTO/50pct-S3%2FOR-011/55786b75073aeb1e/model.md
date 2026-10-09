Let $x_i$ be the number of units of product $i$ to order each day, where $i$ indexes the products in the order given in products.csv.

**Parameters:**

- $v_i$: Value per unit of product $i$ (from "Value" column)
- $w_i$: Weight per unit of product $i$ (from "Weight" column)
- $C$: Overall stock capacity (from "Capacity" in capacity.csv)

**Products (in source order):**

| $i$ | ProductName           | $v_i$ | $w_i$ |
|-----|-----------------------|-------|-------|
| 1   | Spinach               | 64    | 230   |
| 2   | Shiitake Mushrooms    | 75    | 637   |
| 3   | Apples                | 68    | 773   |
| 4   | Carrots               | 11    | 653   |
| 5   | Basil                 | 91    | 755   |
| 6   | Potatoes              | 31    | 670   |
| 7   | Green Beans           | 90    | 505   |
| 8   | Blueberries           | 56    | 821   |
| 9   | Oranges               | 10    | 83    |
| 10  | Watermelons           | 24    | 249   |

Overall stock capacity: $C = 875$

---

**Mathematical Model:**

**Decision variables:**
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,2,\ldots,10\}
$$

**Objective:**
$$
\max \left( 64x_1 + 75x_2 + 68x_3 + 11x_4 + 91x_5 + 31x_6 + 90x_7 + 56x_8 + 10x_9 + 24x_{10} \right)
$$

**Subject to:**
$$
230x_1 + 637x_2 + 773x_3 + 653x_4 + 755x_5 + 670x_6 + 505x_7 + 821x_8 + 83x_9 + 249x_{10} \leq 875
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,2,\ldots,10\}
$$

**Where:**

- $x_1$: Spinach
- $x_2$: Shiitake Mushrooms
- $x_3$: Apples
- $x_4$: Carrots
- $x_5$: Basil
- $x_6$: Potatoes
- $x_7$: Green Beans
- $x_8$: Blueberries
- $x_9$: Oranges
- $x_{10}$: Watermelons

**All coefficients and identifiers are as retrieved from the source files, in original order.**