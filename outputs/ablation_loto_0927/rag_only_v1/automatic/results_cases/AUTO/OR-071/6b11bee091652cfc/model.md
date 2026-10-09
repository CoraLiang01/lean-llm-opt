Let \( x_i \geq 0 \) be the (continuous) number of units to produce of product \( i \), for each product in the order listed in 41.csv. Let the index \( i \) correspond to the row order in the file, with the following parameters for each product \( i \):

- \( L_i \): Labor per unit (from "Labor per unit" column)
- \( M_i \): Material per unit (from "Material per unit" column)
- \( P_i \): Selling Price (from "Selling Price" column)
- \( C_i \): Variable Cost (from "Variable Cost" column)

The model is:

\[
\begin{align*}
\text{Maximize} \quad & \sum_{i=1}^{120} (P_i - C_i)x_i - 4{,}500 \\
\text{subject to} \quad
& \sum_{i=1}^{120} L_i x_i \leq 1{,}650 \\
& \sum_{i=1}^{120} M_i x_i \leq 1{,}850 \\
& x_i \geq 0 \quad \text{for all } i=1,\ldots,120
\end{align*}
\]

Where the products and their coefficients are as follows (in the original file order):

| i | Product Name                        | \( L_i \) | \( M_i \) | \( P_i \) | \( C_i \) |
|---|-------------------------------------|-----------|-----------|-----------|-----------|
| 1 | Classic Oxford Shirt                | 3.1       | 4.2       | 125       | 63        |
| 2 | Cotton Crew Neck T-Shirt            | 2.1       | 3.1       | 83        | 41        |
| 3 | French Terry Hoodie                 | 6.5       | 6.8       | 195       | 95        |
| 4 | Denim Snap Shirt                    | 3.4       | 4.6       | 132       | 68        |
| 5 | Jersey V-Neck T-Shirt               | 2.5       | 3.4       | 90        | 48        |
| 6 | Unisex Jogger Pants                 | 5.9       | 6.1       | 185       | 88        |
| 7 | Flannel Plaid Shirt                 | 3.5       | 4         | 128       | 65        |
| 8 | Performance Polo Shirt              | 2.9       | 3.9       | 118       | 57        |
| 9 | Long-sleeve Henley                  | 2.8       | 3.7       | 95        | 50        |
| 10| Heavyweight Sweatshirt              | 6.2       | 6.4       | 190       | 92        |
| ... | ...                               | ...       | ...       | ...       | ...       |
| 120| Heavyweight Denim Shirt            | 4         | 5.1       | 145       | 77        |

(For brevity, only the first 10 and last product are shown; use all 120 products in the order and with the coefficients as listed in 41.csv.)

Explicitly, the model is:

\[
\begin{align*}
\text{Maximize} \quad & \sum_{i=1}^{120} (P_i - C_i)x_i - 4{,}500 \\
\text{subject to} \quad
& \sum_{i=1}^{120} L_i x_i \leq 1{,}650 \\
& \sum_{i=1}^{120} M_i x_i \leq 1{,}850 \\
& x_i \geq 0 \quad \forall i=1,\ldots,120
\end{align*}
\]

Where all coefficients are taken directly from the corresponding columns in 41.csv, in the original row order. Each \( x_i \) is a nonnegative continuous variable representing the production quantity of product \( i \). The objective is to maximize weekly net profit, defined as total sales revenue minus total variable cost minus the fixed weekly operating cost.