Let $x_i \geq 0$ be the continuous number of units of product $i$ to produce per week, for each product $i$ listed below.

**Parameters (from 41.csv, in source order):**

| Product Name                      | Labor per unit | Material per unit | Selling Price | Variable Cost |
|-----------------------------------|----------------|------------------|--------------|--------------|
| Classic Oxford Shirt              | 3.1            | 4.2              | 125          | 63           |
| Cotton Crew Neck T-Shirt          | 2.1            | 3.1              | 83           | 41           |
| French Terry Hoodie               | 6.5            | 6.8              | 195          | 95           |
| Denim Snap Shirt                  | 3.4            | 4.6              | 132          | 68           |
| Jersey V-Neck T-Shirt             | 2.5            | 3.4              | 90           | 48           |
| Unisex Jogger Pants               | 5.9            | 6.1              | 185          | 88           |
| Flannel Plaid Shirt               | 3.5            | 4                | 128          | 65           |
| Performance Polo Shirt            | 2.9            | 3.9              | 118          | 57           |
| Long-sleeve Henley                | 2.8            | 3.7              | 95           | 50           |
| Heavyweight Sweatshirt            | 6.2            | 6.4              | 190          | 92           |
| Linen Blend Shirt                 | 3.2            | 4.3              | 127          | 64           |
| Graphic Print Tee                 | 2              | 3                | 82           | 40           |
| Quilted Vest                      | 6.8            | 7                | 205          | 105          |
| Chambray Work Shirt               | 3              | 4.1              | 122          | 61           |
| Microfiber Sport Shirt            | 2.3            | 3.3              | 88           | 45           |
| Wool Blend Peacoat                | 7.2            | 7.5              | 215          | 115          |
| Seersucker Short-sleeve           | 2.7            | 3.6              | 93           | 47           |
| Twill Utility Shirt               | 3.3            | 4.5              | 130          | 67           |
| Basic Scoop Neck Tee              | 2.2            | 3.2              | 84           | 42           |
| Waffle Knit Pullover              | 6.1            | 6.3              | 187          | 91           |
| ...                               | ...            | ...              | ...          | ...          |

(Continue for all products as listed in the retrieved data, in the same order.)

**Objective Function:**

Maximize weekly net profit:
\[
\max \left( \sum_{i} ( \text{Selling Price}_i - \text{Variable Cost}_i ) x_i - 4500 \right)
\]

**Subject to:**

1. **Labor Capacity Constraint:**
\[
\sum_{i} (\text{Labor per unit}_i) \cdot x_i \leq 1650
\]

2. **Material Capacity Constraint:**
\[
\sum_{i} (\text{Material per unit}_i) \cdot x_i \leq 1850
\]

3. **Nonnegativity:**
\[
x_i \geq 0 \qquad \forall i
\]

**Where:**

- $x_i$ is the continuous number of units of product $i$ to produce per week.
- All coefficients and product names are as listed in the table above and the full retrieved data, in source order.
- The fixed operating cost of $4,500 is subtracted once in the objective, regardless of production quantities.

**(All products and coefficients must be included as listed in the retrieved data.)**