Let $x_i$ be the nonnegative continuous production quantity of product $i$ (where $i$ indexes the products in the order given in 41.csv).

Parameters (from 41.csv, in source order):

- $l_i$: Labor per unit for product $i$
- $m_i$: Material per unit for product $i$
- $s_i$: Selling price per unit for product $i$
- $v_i$: Variable cost per unit for product $i$

Constants:

- Total weekly labor capacity: $1650$
- Total weekly material capacity: $1850$
- Fixed weekly operating cost: $4500$

Objective:
\[
\max \left( \sum_{i} (s_i - v_i) x_i - 4500 \right)
\]

Subject to:
\[
\sum_{i} l_i x_i \leq 1650
\]
\[
\sum_{i} m_i x_i \leq 1850
\]
\[
x_i \geq 0 \quad \forall i
\]

Where the products and their coefficients are:

| $i$ | Product Name                      | $l_i$ | $m_i$ | $s_i$ | $v_i$ |
|-----|-----------------------------------|-------|-------|-------|-------|
| 1   | Classic Oxford Shirt              | 3.1   | 4.2   | 125   | 63    |
| 2   | Cotton Crew Neck T-Shirt          | 2.1   | 3.1   | 83    | 41    |
| 3   | French Terry Hoodie               | 6.5   | 6.8   | 195   | 95    |
| 4   | Denim Snap Shirt                  | 3.4   | 4.6   | 132   | 68    |
| 5   | Jersey V-Neck T-Shirt             | 2.5   | 3.4   | 90    | 48    |
| 6   | Unisex Jogger Pants               | 5.9   | 6.1   | 185   | 88    |
| 7   | Flannel Plaid Shirt               | 3.5   | 4.0   | 128   | 65    |
| 8   | Performance Polo Shirt            | 2.9   | 3.9   | 118   | 57    |
| 9   | Long-sleeve Henley                | 2.8   | 3.7   | 95    | 50    |
| 10  | Heavyweight Sweatshirt            | 6.2   | 6.4   | 190   | 92    |
| 11  | Linen Blend Shirt                 | 3.2   | 4.3   | 127   | 64    |
| 12  | Graphic Print Tee                 | 2.0   | 3.0   | 82    | 40    |
| 13  | Quilted Vest                      | 6.8   | 7.0   | 205   | 105   |
| 14  | Chambray Work Shirt               | 3.0   | 4.1   | 122   | 61    |
| 15  | Microfiber Sport Shirt            | 2.3   | 3.3   | 88    | 45    |
| 16  | Wool Blend Peacoat                | 7.2   | 7.5   | 215   | 115   |
| 17  | Seersucker Short-sleeve           | 2.7   | 3.6   | 93    | 47    |
| 18  | Twill Utility Shirt               | 3.3   | 4.5   | 130   | 67    |
| 19  | Basic Scoop Neck Tee              | 2.2   | 3.2   | 84    | 42    |
| 20  | Waffle Knit Pullover              | 6.1   | 6.3   | 187   | 91    |
| ... | ...                               | ...   | ...   | ...   | ...   |
| 100 | Heavyweight Denim Shirt           | 4.0   | 5.1   | 145   | 77    |

(Continue for all products in the order and with the coefficients as given in the retrieved data.)

Summary:

\[
\begin{align*}
\max_{x_i \geq 0} \quad & \sum_{i=1}^{100} (s_i - v_i) x_i - 4500 \\
\text{s.t.} \quad & \sum_{i=1}^{100} l_i x_i \leq 1650 \\
                   & \sum_{i=1}^{100} m_i x_i \leq 1850 \\
                   & x_i \geq 0 \quad \forall i=1,\ldots,100
\end{align*}
\]

Where all coefficients and product names are as listed above and in the retrieved data.