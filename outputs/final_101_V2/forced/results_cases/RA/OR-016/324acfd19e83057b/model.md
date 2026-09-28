Let $x_i$ denote the number of units of product $i$ to fulfill, for each product $i$ in the table below.

##### Sets and Parameters

Let $I$ be the set of all products, indexed in the order below:

| $i$ | Product Name              | Revenue $r_i$ | Demand $d_i$ | Initial Inventory $s_i$ |
|-----|--------------------------|---------------|--------------|------------------------|
| 1   | Beauty - 25              | 25            | 240          | 1570                   |
| 2   | Beauty - 30              | 30            | 202          | 1330                   |
| 3   | Beauty - 300             | 300           | 216          | 1420                   |
| 4   | Beauty - 50              | 50            | 263          | 1700                   |
| 5   | Beauty - 500             | 500           | 256          | 1690                   |
| 6   | Clothing - 25            | 25            | 281          | 1840                   |
| 7   | Clothing - 30            | 30            | 261          | 1710                   |
| 8   | Clothing - 300           | 300           | 295          | 1930                   |
| 9   | Clothing - 50            | 50            | 290          | 1890                   |
| 10  | Clothing - 500           | 500           | 244          | 1570                   |
| 11  | Electronics - 25         | 25            | 273          | 1810                   |
| 12  | Electronics - 30         | 30            | 220          | 1410                   |
| 13  | Electronics - 300        | 300           | 286          | 1830                   |
| 14  | Electronics - 50         | 50            | 268          | 1750                   |
| 15  | Electronics - 500        | 500           | 262          | 1690                   |
| 16  | Home Goods - 25          | 25            | 255          | 1660                   |
| 17  | Home Goods - 30          | 30            | 218          | 1417                   |
| 18  | Home Goods - 50          | 50            | 278          | 1807                   |
| 19  | Home Goods - 300         | 300           | 195          | 1268                   |
| 20  | Home Goods - 500         | 500           | 248          | 1612                   |
| 21  | Sports - 25              | 25            | 269          | 1749                   |
| 22  | Sports - 30              | 30            | 227          | 1476                   |
| 23  | Sports - 50              | 50            | 285          | 1853                   |
| 24  | Sports - 300             | 300           | 208          | 1352                   |
| 25  | Sports - 500             | 500           | 259          | 1684                   |
| 26  | Furniture - 25           | 25            | 242          | 1573                   |
| 27  | Furniture - 30           | 30            | 213          | 1385                   |
| 28  | Furniture - 50           | 50            | 272          | 1768                   |
| 29  | Furniture - 300          | 300           | 224          | 1456                   |
| 30  | Furniture - 500          | 500           | 251          | 1632                   |
| 31  | Toys - 25                | 25            | 257          | 1671                   |
| 32  | Toys - 30                | 30            | 235          | 1528                   |
| 33  | Toys - 50                | 50            | 291          | 1892                   |
| 34  | Toys - 300               | 300           | 199          | 1294                   |
| 35  | Toys - 500               | 500           | 264          | 1716                   |

##### Decision Variables

For each product $i \in I$:
- $x_i$: Number of units of product $i$ to fulfill (allocate to demand), $x_i \in \mathbb{Z}_{\geq 0}$

##### Objective

Maximize total revenue:
$$
\max \sum_{i \in I} r_i x_i
$$

##### Constraints

For each product $i \in I$:
- Demand fulfillment cannot exceed demand:
  $$
  x_i \leq d_i
  $$
- Fulfillment cannot exceed available inventory:
  $$
  x_i \leq s_i
  $$
- Nonnegativity and integrality:
  $$
  x_i \in \mathbb{Z}_{\geq 0}
  $$

##### Complete Model

$$
\begin{align*}
\max \quad & \sum_{i \in I} r_i x_i \\
\text{s.t.} \quad & x_i \leq d_i, \quad \forall i \in I \\
                  & x_i \leq s_i, \quad \forall i \in I \\
                  & x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\end{align*}
$$

Where all parameters ($r_i$, $d_i$, $s_i$) and product names are as listed above, in the original order.