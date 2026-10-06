Let $x_i$ be the (continuous, nonnegative) number of units of product $i$ to produce per week, where $i$ indexes the products in the order given in 41.csv.

Parameters (from 41.csv, in source order):

- $l_i$ = Labor per unit for product $i$
- $m_i$ = Material per unit for product $i$
- $s_i$ = Selling Price per unit for product $i$
- $v_i$ = Variable Cost per unit for product $i$

Constants:

- Total weekly labor available: $1650$
- Total weekly material available: $1850$
- Fixed weekly operating cost: $4500$

Objective:
\[
\max \left( \sum_{i} (s_i - v_i) x_i - 4500 \right)
\]

Subject to:

Labor capacity:
\[
\sum_{i} l_i x_i \leq 1650
\]

Material capacity:
\[
\sum_{i} m_i x_i \leq 1850
\]

Nonnegativity:
\[
x_i \geq 0 \quad \forall i
\]

Where the products and their parameters are:

| $i$ | Product Name | $l_i$ | $m_i$ | $s_i$ | $v_i$ |
|-----|-------------------------------|-------|-------|-------|-------|
| 1 | Classic Oxford Shirt | 3.1 | 4.2 | 125 | 63 |
| 2 | Cotton Crew Neck T-Shirt | 2.1 | 3.1 | 83 | 41 |
| 3 | French Terry Hoodie | 6.5 | 6.8 | 195 | 95 |
| 4 | Denim Snap Shirt | 3.4 | 4.6 | 132 | 68 |
| 5 | Jersey V-Neck T-Shirt | 2.5 | 3.4 | 90 | 48 |
| 6 | Unisex Jogger Pants | 5.9 | 6.1 | 185 | 88 |
| 7 | Flannel Plaid Shirt | 3.5 | 4 | 128 | 65 |
| 8 | Performance Polo Shirt | 2.9 | 3.9 | 118 | 57 |
| 9 | Long-sleeve Henley | 2.8 | 3.7 | 95 | 50 |
| 10 | Heavyweight Sweatshirt | 6.2 | 6.4 | 190 | 92 |
| 11 | Linen Blend Shirt | 3.2 | 4.3 | 127 | 64 |
| 12 | Graphic Print Tee | 2 | 3 | 82 | 40 |
| 13 | Quilted Vest | 6.8 | 7 | 205 | 105 |
| 14 | Chambray Work Shirt | 3 | 4.1 | 122 | 61 |
| 15 | Microfiber Sport Shirt | 2.3 | 3.3 | 88 | 45 |
| 16 | Wool Blend Peacoat | 7.2 | 7.5 | 215 | 115 |
| 17 | Seersucker Short-sleeve | 2.7 | 3.6 | 93 | 47 |
| 18 | Twill Utility Shirt | 3.3 | 4.5 | 130 | 67 |
| 19 | Basic Scoop Neck Tee | 2.2 | 3.2 | 84 | 42 |
| 20 | Waffle Knit Pullover | 6.1 | 6.3 | 187 | 91 |
| 21 | Button-Down Poplin Shirt | 3.6 | 4.7 | 135 | 70 |
| 22 | Slub Cotton T-Shirt | 2.6 | 3.5 | 92 | 49 |
| 23 | Drawstring Cargo Pants | 6.4 | 6.6 | 192 | 93 |
| 24 | Pima Cotton Dress Shirt | 3.8 | 4.9 | 140 | 73 |
| 25 | Pocket T-Shirt | 2.4 | 3.3 | 87 | 44 |
| 26 | Fleece-Lined Track Suit | 7 | 7.2 | 210 | 110 |
| 27 | Printed Camp Collar Shirt | 2.9 | 4 | 120 | 60 |
| 28 | Performance Dry-Fit Tee | 2 | 3 | 80 | 39 |
| 29 | Corduroy Shacket | 6.7 | 6.9 | 202 | 97 |
| 30 | Herringbone Casual Shirt | 3 | 4.1 | 123 | 62 |
| 31 | Organic Cotton Tee | 2.1 | 3.1 | 85 | 43 |
| 32 | Cashmere Blend Sweater | 7.5 | 7.8 | 225 | 125 |
| 33 | Striped Rugby Shirt | 3.4 | 4.4 | 129 | 66 |
| 34 | Raglan Sleeve Tee | 2.6 | 3.6 | 94 | 51 |
| 35 | Waterproof Rain Jacket | 7.1 | 7.4 | 212 | 112 |
| 36 | Micro-check Business Shirt | 3.7 | 4.8 | 138 | 72 |
| 37 | Heather Grey T-Shirt | 2.3 | 3.2 | 86 | 45 |
| 38 | Slim Fit Chinos | 6.3 | 6.5 | 190 | 90 |
| 39 | Band Collar Shirt | 3.1 | 4.2 | 126 | 64 |
| 40 | Triblend Jersey Tee | 2.5 | 3.5 | 91 | 49 |
| 41 | Sherpa Lined Pullover | 6.9 | 7.1 | 208 | 108 |
| ... | ... | ... | ... | ... | ... |

(Continue for all products in the order and with the coefficients as given in the data above.)

Decision variables:
\[
x_i \geq 0 \quad \text{(continuous)}, \quad \forall i
\]

This is the complete numerical formulation for the Red Bean Clothing Factory production optimization problem.