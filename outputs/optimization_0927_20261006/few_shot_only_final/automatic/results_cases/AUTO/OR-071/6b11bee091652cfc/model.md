Let $x_i \geq 0$ be the production quantity (in units) of product $i$, for each product $i$ in the set of all products listed in 41.csv.

Let:
- $L_i$ = Labor per unit for product $i$
- $M_i$ = Material per unit for product $i$
- $P_i$ = Selling Price for product $i$
- $C_i$ = Variable Cost for product $i$

**Objective Function:**

\[
\max \left( \sum_{i} (P_i - C_i) x_i - 4500 \right)
\]

**Subject to:**

\[
\sum_{i} L_i x_i \leq 1650
\]
\[
\sum_{i} M_i x_i \leq 1850
\]
\[
x_i \geq 0 \quad \forall i
\]

**Where:**

For each product $i$ (using the original order and identifiers from 41.csv):

| Product Name                      | $L_i$ (Labor per unit) | $M_i$ (Material per unit) | $P_i$ (Selling Price) | $C_i$ (Variable Cost) |
|------------------------------------|------------------------|---------------------------|-----------------------|-----------------------|
| Classic Oxford Shirt               | 3.1                    | 4.2                       | 125                   | 63                    |
| Cotton Crew Neck T-Shirt           | 2.1                    | 3.1                       | 83                    | 41                    |
| French Terry Hoodie                | 6.5                    | 6.8                       | 195                   | 95                    |
| Denim Snap Shirt                   | 3.4                    | 4.6                       | 132                   | 68                    |
| Jersey V-Neck T-Shirt              | 2.5                    | 3.4                       | 90                    | 48                    |
| Unisex Jogger Pants                | 5.9                    | 6.1                       | 185                   | 88                    |
| Flannel Plaid Shirt                | 3.5                    | 4                         | 128                   | 65                    |
| Performance Polo Shirt             | 2.9                    | 3.9                       | 118                   | 57                    |
| Long-sleeve Henley                 | 2.8                    | 3.7                       | 95                    | 50                    |
| Heavyweight Sweatshirt             | 6.2                    | 6.4                       | 190                   | 92                    |
| Linen Blend Shirt                  | 3.2                    | 4.3                       | 127                   | 64                    |
| Graphic Print Tee                  | 2                      | 3                         | 82                    | 40                    |
| Quilted Vest                       | 6.8                    | 7                         | 205                   | 105                   |
| Chambray Work Shirt                | 3                      | 4.1                       | 122                   | 61                    |
| Microfiber Sport Shirt             | 2.3                    | 3.3                       | 88                    | 45                    |
| Wool Blend Peacoat                 | 7.2                    | 7.5                       | 215                   | 115                   |
| Seersucker Short-sleeve            | 2.7                    | 3.6                       | 93                    | 47                    |
| Twill Utility Shirt                | 3.3                    | 4.5                       | 130                   | 67                    |
| Basic Scoop Neck Tee               | 2.2                    | 3.2                       | 84                    | 42                    |
| Waffle Knit Pullover               | 6.1                    | 6.3                       | 187                   | 91                    |
| Button-Down Poplin Shirt           | 3.6                    | 4.7                       | 135                   | 70                    |
| Slub Cotton T-Shirt                | 2.6                    | 3.5                       | 92                    | 49                    |
| Drawstring Cargo Pants             | 6.4                    | 6.6                       | 192                   | 93                    |
| Pima Cotton Dress Shirt            | 3.8                    | 4.9                       | 140                   | 73                    |
| Pocket T-Shirt                     | 2.4                    | 3.3                       | 87                    | 44                    |
| Fleece-Lined Track Suit            | 7                      | 7.2                       | 210                   | 110                   |
| Printed Camp Collar Shirt          | 2.9                    | 4                         | 120                   | 60                    |
| Performance Dry-Fit Tee            | 2                      | 3                         | 80                    | 39                    |
| Corduroy Shacket                   | 6.7                    | 6.9                       | 202                   | 97                    |
| Herringbone Casual Shirt           | 3                      | 4.1                       | 123                   | 62                    |
| ...                                | ...                    | ...                       | ...                   | ...                   |
| (Continue for all 198 products in original order) |

**Decision variables:**

\[
x_i \geq 0 \quad \text{(continuous)}, \quad \forall i \in \{\text{all products in 41.csv, in original order}\}
\]

**All coefficients and product names are as listed in the original data.**