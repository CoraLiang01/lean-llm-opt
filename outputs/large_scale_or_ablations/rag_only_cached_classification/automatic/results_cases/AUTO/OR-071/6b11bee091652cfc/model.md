Let \( x_i \geq 0 \) be the (continuous) number of units of product \( i \) to produce per week, for each product \( i \) in the order given in the table below.

Let the set of products and their parameters be as follows (in original file order):

| Index | Product Name                       | Labor per unit | Material per unit | Selling Price | Variable Cost |
|-------|------------------------------------|----------------|------------------|--------------|--------------|
| 1     | Classic Oxford Shirt               | 3.1            | 4.2              | 125          | 63           |
| 2     | Cotton Crew Neck T-Shirt           | 2.1            | 3.1              | 83           | 41           |
| 3     | French Terry Hoodie                | 6.5            | 6.8              | 195          | 95           |
| 4     | Denim Snap Shirt                   | 3.4            | 4.6              | 132          | 68           |
| 5     | Jersey V-Neck T-Shirt              | 2.5            | 3.4              | 90           | 48           |
| ...   | ...                                | ...            | ...              | ...          | ...          |
| 100   | Heavyweight Denim Shirt            | 4.0            | 5.1              | 145          | 77           |

(Full list as in the CSV; for brevity, only the first five and last row are shown here, but all 100 products are included in the model.)

Define:
- \( x_i \geq 0 \) (continuous), for \( i = 1, \ldots, 100 \), the production quantity of product \( i \).

Parameters (from the table above, for each \( i \)):
- \( \ell_i \): Labor per unit for product \( i \)
- \( m_i \): Material per unit for product \( i \)
- \( p_i \): Selling price per unit for product \( i \)
- \( c_i \): Variable cost per unit for product \( i \)

Constants:
- Weekly labor capacity: 1,650 units
- Weekly material capacity: 1,850 units
- Fixed weekly operating cost: \$4,500

The optimization model is:

\[
\begin{align*}
\text{Maximize} \quad & \sum_{i=1}^{100} (p_i - c_i) x_i - 4500 \\
\text{subject to} \quad
& \sum_{i=1}^{100} \ell_i x_i \leq 1650 \\
& \sum_{i=1}^{100} m_i x_i \leq 1850 \\
& x_i \geq 0 \quad \text{for all } i = 1, \ldots, 100
\end{align*}
\]

Where:
- \( (p_i - c_i) \) is the net profit per unit of product \( i \).
- The first constraint ensures total labor used does not exceed 1,650 units per week.
- The second constraint ensures total material used does not exceed 1,850 units per week.
- The objective subtracts the fixed weekly operating cost of \$4,500 from total net profit.

All coefficients (\( \ell_i, m_i, p_i, c_i \)) are taken directly from the corresponding columns of 41.csv, in the original row order.

This is a continuous linear program with 100 variables and 2 constraints (plus nonnegativity).