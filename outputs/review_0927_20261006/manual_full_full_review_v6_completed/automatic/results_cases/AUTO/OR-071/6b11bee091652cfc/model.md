Let $x_i$ be the nonnegative continuous production quantity of product $i$ (where $i$ indexes the products in the table below).

**Parameters (from 41.csv):**

| Product Name                      | Labor per unit | Material per unit | Selling Price | Variable Cost |
|-----------------------------------|:--------------:|:-----------------:|:-------------:|:-------------:|
| Classic Oxford Shirt              | 3.1            | 4.2               | 125           | 63            |
| Cotton Crew Neck T-Shirt          | 2.1            | 3.1               | 83            | 41            |
| French Terry Hoodie               | 6.5            | 6.8               | 195           | 95            |
| Denim Snap Shirt                  | 3.4            | 4.6               | 132           | 68            |
| Jersey V-Neck T-Shirt             | 2.5            | 3.4               | 90            | 48            |
| Unisex Jogger Pants               | 5.9            | 6.1               | 185           | 88            |
| Flannel Plaid Shirt               | 3.5            | 4                 | 128           | 65            |
| Performance Polo Shirt            | 2.9            | 3.9               | 118           | 57            |
| Long-sleeve Henley                | 2.8            | 3.7               | 95            | 50            |
| Heavyweight Sweatshirt            | 6.2            | 6.4               | 190           | 92            |
| Linen Blend Shirt                 | 3.2            | 4.3               | 127           | 64            |
| Graphic Print Tee                 | 2              | 3                 | 82            | 40            |
| Quilted Vest                      | 6.8            | 7                 | 205           | 105           |
| Chambray Work Shirt               | 3              | 4.1               | 122           | 61            |
| Microfiber Sport Shirt            | 2.3            | 3.3               | 88            | 45            |
| Wool Blend Peacoat                | 7.2            | 7.5               | 215           | 115           |
| Seersucker Short-sleeve           | 2.7            | 3.6               | 93            | 47            |
| Twill Utility Shirt               | 3.3            | 4.5               | 130           | 67            |
| Basic Scoop Neck Tee              | 2.2            | 3.2               | 84            | 42            |
| Waffle Knit Pullover              | 6.1            | 6.3               | 187           | 91            |
| ...                               | ...            | ...               | ...           | ...           |

(Continue for all products as listed in the retrieved data.)

**Fixed weekly operating cost:** $4500$

**Labor capacity:** $1650$  
**Material capacity:** $1850$

---

### Mathematical Model

**Decision variables:**  
For each product $i$, $x_i \geq 0$ (continuous), the number of units of product $i$ to produce per week.

#### Objective:
Maximize weekly net profit:
\[
\max \left( \sum_{i} (p_i - v_i) x_i - 4500 \right)
\]
where:
- $p_i$ = Selling Price of product $i$
- $v_i$ = Variable Cost of product $i$

#### Constraints:

1. **Labor capacity:**
   \[
   \sum_{i} a_i x_i \leq 1650
   \]
   where $a_i$ = Labor per unit for product $i$

2. **Material capacity:**
   \[
   \sum_{i} b_i x_i \leq 1850
   \]
   where $b_i$ = Material per unit for product $i$

3. **Nonnegativity:**
   \[
   x_i \geq 0 \quad \forall i
   \]

---

#### Explicit Formulation (with all products):

Let $I$ be the set of all products in the table above. For each $i \in I$:

- $a_i$ = Labor per unit (from "Labor per unit" column)
- $b_i$ = Material per unit (from "Material per unit" column)
- $p_i$ = Selling Price
- $v_i$ = Variable Cost

\[
\begin{align*}
\max_{x_i \geq 0} \quad & \sum_{i \in I} (p_i - v_i) x_i - 4500 \\
\text{s.t.} \quad & \sum_{i \in I} a_i x_i \leq 1650 \\
                   & \sum_{i \in I} b_i x_i \leq 1850 \\
                   & x_i \geq 0 \quad \forall i \in I
\end{align*}
\]

All coefficients and product names are as given in the retrieved data.