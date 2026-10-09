Let $x_i$ be the number of units of product $i$ to order each day, where $i$ indexes the products in the order given in 'products.csv'.

**Parameters:**

- For $i=1,\ldots,10$ (in the order of the table below):

| $i$ | ProductName           | Weight ($w_i$) | Value ($v_i$) |
|-----|----------------------|---------------|--------------|
| 1   | Spinach              | 230           | 64           |
| 2   | Shiitake Mushrooms   | 637           | 75           |
| 3   | Apples               | 773           | 68           |
| 4   | Carrots              | 653           | 11           |
| 5   | Basil                | 755           | 91           |
| 6   | Potatoes             | 670           | 31           |
| 7   | Green Beans          | 505           | 90           |
| 8   | Blueberries          | 821           | 56           |
| 9   | Oranges              | 83            | 10           |
| 10  | Watermelons          | 249           | 24           |

- Total stock capacity: $C = 875$

**Decision variables:**

- $x_i \in \mathbb{Z}_{\geq 0}$, for $i=1,\ldots,10$

**Mathematical Model:**

Maximize total value:
$$
\max \sum_{i=1}^{10} v_i x_i
$$

Subject to the overall stock capacity:
$$
\sum_{i=1}^{10} w_i x_i \leq 875
$$

Nonnegativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i=1,\ldots,10
$$

Where:

- $v_i$ and $w_i$ are as given in the table above for each product $i$.
- $C = 875$ is the total stock capacity.