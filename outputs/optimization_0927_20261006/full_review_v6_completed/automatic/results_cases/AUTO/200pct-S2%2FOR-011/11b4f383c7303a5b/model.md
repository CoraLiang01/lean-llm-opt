Let $x_i$ be the number of units of product $i$ to order each day, where $i$ indexes the products listed below.

**Parameters:**

- For each product $i$:
    - $v_i$ = Value (from products.csv)
    - $w_i$ = Weight (from products.csv)
    - ProductName as below

- $C$ = 875 (overall stock capacity from capacity.csv)

**Products (in source order):**

| $i$ | ProductName           | $v_i$ (Value) | $w_i$ (Weight) |
|-----|-----------------------|---------------|---------------|
| 1   | Spinach               | 64            | 230           |
| 2   | Shiitake Mushrooms    | 75            | 637           |
| 3   | Apples                | 68            | 773           |
| 4   | Carrots               | 11            | 653           |
| 5   | Basil                 | 91            | 755           |
| 6   | Potatoes              | 31            | 670           |
| 7   | Green Beans           | 90            | 505           |
| 8   | Blueberries           | 56            | 821           |
| 9   | Oranges               | 10            | 83            |
| 10  | Watermelons           | 24            | 249           |

**Decision Variables:**

$x_i \in \mathbb{Z}_{\geq 0}$, for $i = 1, \ldots, 10$

---

**Mathematical Model:**

Objective:
$$
\max \sum_{i=1}^{10} v_i x_i
$$

Subject to:
$$
\sum_{i=1}^{10} w_i x_i \leq 875
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 10
$$

---

**Parameter Values (from retrieved data):**

- $v_1 = 64$, $w_1 = 230$  (Spinach)
- $v_2 = 75$, $w_2 = 637$  (Shiitake Mushrooms)
- $v_3 = 68$, $w_3 = 773$  (Apples)
- $v_4 = 11$, $w_4 = 653$  (Carrots)
- $v_5 = 91$, $w_5 = 755$  (Basil)
- $v_6 = 31$, $w_6 = 670$  (Potatoes)
- $v_7 = 90$, $w_7 = 505$  (Green Beans)
- $v_8 = 56$, $w_8 = 821$  (Blueberries)
- $v_9 = 10$, $w_9 = 83$  (Oranges)
- $v_{10} = 24$, $w_{10} = 249$ (Watermelons)

- $C = 875$

---

**Complete Model:**

$$
\max\ 64x_1 + 75x_2 + 68x_3 + 11x_4 + 91x_5 + 31x_6 + 90x_7 + 56x_8 + 10x_9 + 24x_{10}
$$

$$
230x_1 + 637x_2 + 773x_3 + 653x_4 + 755x_5 + 670x_6 + 505x_7 + 821x_8 + 83x_9 + 249x_{10} \leq 875
$$

$$
x_i \in \mathbb{Z}_{\geq 0},\quad i=1,\ldots,10
$$

Where $x_i$ is the number of units of product $i$ to order each day, as listed above.