##### Decision Variables

Let $x_i$ denote the fulfillment quantity for product $i$, where $i$ indexes all products across all categories.

##### Parameters

For each product $i$:
- $r_i$: Revenue per unit
- $s_i$: Initial Inventory
- $d_i$: Demand

The products and their parameters are as follows:

| Product Name              | $r_i$ | $s_i$ | $d_i$ |
|--------------------------|-------|-------|-------|
| Beauty - 25              | 25    | 1570  | 240   |
| Beauty - 30              | 30    | 1330  | 202   |
| Beauty - 50              | 50    | 1700  | 263   |
| Beauty - 300             | 300   | 1420  | 216   |
| Beauty - 500             | 500   | 1690  | 256   |
| Clothing - 25            | 25    | 1840  | 281   |
| Clothing - 30            | 30    | 1710  | 261   |
| Clothing - 50            | 50    | 1890  | 290   |
| Clothing - 300           | 300   | 1930  | 295   |
| Clothing - 500           | 500   | 1570  | 244   |
| Electronics - 25         | 25    | 1810  | 273   |
| Electronics - 30         | 30    | 1410  | 220   |
| Electronics - 50         | 50    | 1750  | 268   |
| Electronics - 300        | 300   | 1830  | 286   |
| Electronics - 500        | 500   | 1690  | 262   |
| Home Goods - 25          | 25    | 1660  | 255   |
| Home Goods - 30          | 30    | 1417  | 218   |
| Home Goods - 50          | 50    | 1807  | 278   |
| Home Goods - 300         | 300   | 1268  | 195   |
| Home Goods - 500         | 500   | 1612  | 248   |
| Sports - 25              | 25    | 1749  | 269   |
| Sports - 30              | 30    | 1476  | 227   |
| Sports - 50              | 50    | 1853  | 285   |
| Sports - 300             | 300   | 1352  | 208   |
| Sports - 500             | 500   | 1684  | 259   |
| Furniture - 25           | 25    | 1573  | 242   |
| Furniture - 30           | 30    | 1385  | 213   |
| Furniture - 50           | 50    | 1768  | 272   |
| Furniture - 300          | 300   | 1456  | 224   |
| Furniture - 500          | 500   | 1632  | 251   |
| Toys - 25                | 25    | 1671  | 257   |
| Toys - 30                | 30    | 1528  | 235   |
| Toys - 50                | 50    | 1892  | 291   |
| Toys - 300               | 300   | 1294  | 199   |
| Toys - 500               | 500   | 1716  | 264   |

##### Objective Function

$\max \sum_{i} r_i x_i$

##### Constraints

For each product $i$:
- Inventory and demand limits:
  $$
  0 \leq x_i \leq \min\{s_i, d_i\}
  $$

##### Variable Domains

$x_i$ are continuous variables (if partial fulfillment is allowed), or integer variables (if only whole units can be fulfilled), for all $i$.

##### Retrieved Information

{
  "products": [
    {"name": "Beauty - 25", "revenue": 25, "initial_inventory": 1570, "demand": 240},
    {"name": "Beauty - 30", "revenue": 30, "initial_inventory": 1330, "demand": 202},
    {"name": "Beauty - 50", "revenue": 50, "initial_inventory": 1700, "demand": 263},
    {"name": "Beauty - 300", "revenue": 300, "initial_inventory": 1420, "demand": 216},
    {"name": "Beauty - 500", "revenue": 500, "initial_inventory": 1690, "demand": 256},
    {"name": "Clothing - 25", "revenue": 25, "initial_inventory": 1840, "demand": 281},
    {"name": "Clothing - 30", "revenue": 30, "initial_inventory": 1710, "demand": 261},
    {"name": "Clothing - 50", "revenue": 50, "initial_inventory": 1890, "demand": 290},
    {"name": "Clothing - 300", "revenue": 300, "initial_inventory": 1930, "demand": 295},
    {"name": "Clothing - 500", "revenue": 500, "initial_inventory": 1570, "demand": 244},
    {"name": "Electronics - 25", "revenue": 25, "initial_inventory": 1810, "demand": 273},
    {"name": "Electronics - 30", "revenue": 30, "initial_inventory": 1410, "demand": 220},
    {"name": "Electronics - 50", "revenue": 50, "initial_inventory": 1750, "demand": 268},
    {"name": "Electronics - 300", "revenue": 300, "initial_inventory": 1830, "demand": 286},
    {"name": "Electronics - 500", "revenue": 500, "initial_inventory": 1690, "demand": 262},
    {"name": "Home Goods - 25", "revenue": 25, "initial_inventory": 1660, "demand": 255},
    {"name": "Home Goods - 30", "revenue": 30, "initial_inventory": 1417, "demand": 218},
    {"name": "Home Goods - 50", "revenue": 50, "initial_inventory": 1807, "demand": 278},
    {"name": "Home Goods - 300", "revenue": 300, "initial_inventory": 1268, "demand": 195},
    {"name": "Home Goods - 500", "revenue": 500, "initial_inventory": 1612, "demand": 248},
    {"name": "Sports - 25", "revenue": 25, "initial_inventory": 1749, "demand": 269},
    {"name": "Sports - 30", "revenue": 30, "initial_inventory": 1476, "demand": 227},
    {"name": "Sports - 50", "revenue": 50, "initial_inventory": 1853, "demand": 285},
    {"name": "Sports - 300", "revenue": 300, "initial_inventory": 1352, "demand": 208},
    {"name": "Sports - 500", "revenue": 500, "initial_inventory": 1684, "demand": 259},
    {"name": "Furniture - 25", "revenue": 25, "initial_inventory": 1573, "demand": 242},
    {"name": "Furniture - 30", "revenue": 30, "initial_inventory": 1385, "demand": 213},
    {"name": "Furniture - 50", "revenue": 50, "initial_inventory": 1768, "demand": 272},
    {"name": "Furniture - 300", "revenue": 300, "initial_inventory": 1456, "demand": 224},
    {"name": "Furniture - 500", "revenue": 500, "initial_inventory": 1632, "demand": 251},
    {"name": "Toys - 25", "revenue": 25, "initial_inventory": 1671, "demand": 257},
    {"name": "Toys - 30", "revenue": 30, "initial_inventory": 1528, "demand": 235},
    {"name": "Toys - 50", "revenue": 50, "initial_inventory": 1892, "demand": 291},
    {"name": "Toys - 300", "revenue": 300, "initial_inventory": 1294, "demand": 199},
    {"name": "Toys - 500", "revenue": 500, "initial_inventory": 1716, "demand": 264}
  ]
}

##### Complete Mathematical Model

$\boxed{
\begin{align*}
\max_{x_i} \quad & \sum_{i} r_i x_i \\
\text{s.t.} \quad & 0 \leq x_i \leq \min\{s_i, d_i\} \quad \forall i \\
& x_i \in \mathbb{R}_+ \quad \text{(or } x_i \in \mathbb{Z}_+ \text{ if integer units required)}
\end{align*}
}$

where $i$ indexes all products as listed above, with their respective $r_i$, $s_i$, and $d_i$ values.