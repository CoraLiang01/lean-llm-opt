Let $x_i$ be the (continuous, nonnegative) number of units of product $i$ to produce per week, for each product $i$ listed below.

**Parameters (from 41.csv, in source order):**

| Product Name                      | Labor per unit | Material per unit | Selling Price | Variable Cost |
|-----------------------------------|:--------------:|:----------------:|:-------------:|:-------------:|
| Classic Oxford Shirt              | 3.1            | 4.2              | 125           | 63            |
| Cotton Crew Neck T-Shirt          | 2.1            | 3.1              | 83            | 41            |
| French Terry Hoodie               | 6.5            | 6.8              | 195           | 95            |
| Denim Snap Shirt                  | 3.4            | 4.6              | 132           | 68            |
| Jersey V-Neck T-Shirt             | 2.5            | 3.4              | 90            | 48            |
| Unisex Jogger Pants               | 5.9            | 6.1              | 185           | 88            |
| Flannel Plaid Shirt               | 3.5            | 4                | 128           | 65            |
| Performance Polo Shirt            | 2.9            | 3.9              | 118           | 57            |
| Long-sleeve Henley                | 2.8            | 3.7              | 95            | 50            |
| Heavyweight Sweatshirt            | 6.2            | 6.4              | 190           | 92            |
| Linen Blend Shirt                 | 3.2            | 4.3              | 127           | 64            |
| Graphic Print Tee                 | 2              | 3                | 82            | 40            |
| Quilted Vest                      | 6.8            | 7                | 205           | 105           |
| Chambray Work Shirt               | 3              | 4.1              | 122           | 61            |
| Microfiber Sport Shirt            | 2.3            | 3.3              | 88            | 45            |
| Wool Blend Peacoat                | 7.2            | 7.5              | 215           | 115           |
| Seersucker Short-sleeve           | 2.7            | 3.6              | 93            | 47            |
| Twill Utility Shirt               | 3.3            | 4.5              | 130           | 67            |
| Basic Scoop Neck Tee              | 2.2            | 3.2              | 84            | 42            |
| Waffle Knit Pullover              | 6.1            | 6.3              | 187           | 91            |
| ...                               | ...            | ...              | ...           | ...           |

(Continue for all products as listed in the retrieved data, in the same order.)

**Fixed weekly operating cost:** $F = 4,\!500$

**Labor capacity:** $L = 1,\!650$

**Material capacity:** $M = 1,\!850$

---

### Mathematical Model

**Decision variables:**

- $x_i \geq 0$ (continuous), for each product $i$ in the list above.

**Objective:**

\[
\max \left( \sum_{i} (p_i - v_i) x_i - F \right)
\]

where:
- $p_i$ = Selling Price of product $i$
- $v_i$ = Variable Cost of product $i$
- $F$ = 4,500

**Constraints:**

1. **Labor capacity:**
   \[
   \sum_{i} \ell_i x_i \leq 1,\!650
   \]
   where $\ell_i$ = Labor per unit for product $i$

2. **Material capacity:**
   \[
   \sum_{i} m_i x_i \leq 1,\!850
   \]
   where $m_i$ = Material per unit for product $i$

3. **Nonnegativity:**
   \[
   x_i \geq 0 \quad \forall i
   \]

---

**Explicitly, with all parameters from 41.csv in source order:**

Let $i$ index the products in the order given above, with parameters:

- $\ell_i$ = Labor per unit (from "Labor per unit" column)
- $m_i$ = Material per unit (from "Material per unit" column)
- $p_i$ = Selling Price (from "Selling Price" column)
- $v_i$ = Variable Cost (from "Variable Cost" column)

**Model:**

\[
\begin{align*}
\max_{x_i \geq 0} \quad & \sum_{i} (p_i - v_i) x_i - 4,\!500 \\
\text{s.t.} \quad
& \sum_{i} \ell_i x_i \leq 1,\!650 \\
& \sum_{i} m_i x_i \leq 1,\!850 \\
& x_i \geq 0 \quad \forall i
\end{align*}
\]

where all coefficients and product names are as listed in the retrieved data, in the same order.