Let $x_i$ be the number of units of product $i$ to order each day, where $i$ indexes the products as listed below.

**Products and Parameters (in source order):**

| $i$ | ProductName           | Value ($v_i$) | Weight ($w_i$) |
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

Total stock capacity: $C = 875$

---

**Mathematical Model:**

**Objective:**
\[
\max \; 64x_1 + 75x_2 + 68x_3 + 11x_4 + 91x_5 + 31x_6 + 90x_7 + 56x_8 + 10x_9 + 24x_{10}
\]

**Subject to:**
\[
230x_1 + 637x_2 + 773x_3 + 653x_4 + 755x_5 + 670x_6 + 505x_7 + 821x_8 + 83x_9 + 249x_{10} \leq 875
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1,2,\ldots,10
\]

**Where:**
- $x_i$ = number of units of product $i$ to order each day (nonnegative integer)
- $v_i$ = Value of product $i$ (see table above)
- $w_i$ = Weight of product $i$ (see table above)
- $C$ = total stock capacity (875)

**All coefficients and identifiers are as retrieved and in original order.**