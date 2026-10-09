#### Sets and Indices

Let $i$ index the products, with the following mapping:

| $i$ | ProductName           |
|-----|-----------------------|
| 1   | Spinach               |
| 2   | Shiitake Mushrooms    |
| 3   | Apples                |
| 4   | Carrots               |
| 5   | Basil                 |
| 6   | Potatoes              |
| 7   | Green Beans           |
| 8   | Blueberries           |
| 9   | Oranges               |
| 10  | Watermelons           |

#### Parameters

| ProductName         | Value ($v_i$) | Weight ($w_i$) |
|---------------------|--------------|---------------|
| Spinach             | 64           | 230           |
| Shiitake Mushrooms  | 75           | 637           |
| Apples              | 68           | 773           |
| Carrots             | 11           | 653           |
| Basil               | 91           | 755           |
| Potatoes            | 31           | 670           |
| Green Beans         | 90           | 505           |
| Blueberries         | 56           | 821           |
| Oranges             | 10           | 83            |
| Watermelons         | 24           | 249           |

Total stock capacity: $C = 875$

#### Decision Variables

$x_i$ = number of units of product $i$ to order each day, $x_i \in \mathbb{Z}_{\geq 0}$

#### Mathematical Model

**Objective:**
\[
\max \sum_{i=1}^{10} v_i x_i
\]
where $v_i$ is the Value for product $i$.

**Constraint:**
\[
\sum_{i=1}^{10} w_i x_i \leq 875
\]
where $w_i$ is the Weight for product $i$.

**Variable Domains:**
\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 10
\]

#### Parameter Table (from retrieved data)

| ProductName         | Value | Weight |
|---------------------|-------|--------|
| Spinach             | 64    | 230    |
| Shiitake Mushrooms  | 75    | 637    |
| Apples              | 68    | 773    |
| Carrots             | 11    | 653    |
| Basil               | 91    | 755    |
| Potatoes            | 31    | 670    |
| Green Beans         | 90    | 505    |
| Blueberries         | 56    | 821    |
| Oranges             | 10    | 83     |
| Watermelons         | 24    | 249    |

**Total stock capacity:** $875$