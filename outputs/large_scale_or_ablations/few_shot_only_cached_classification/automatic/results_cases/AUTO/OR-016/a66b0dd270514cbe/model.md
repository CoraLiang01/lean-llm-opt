Let $i$ index all products in the dataset, in the source order below. Let $x_i$ be the number of units fulfilled for product $i$.

Objective:
$$
\max \sum_{i} A_i \cdot x_i
$$
where $A_i$ is the Revenue for product $i$.

Subject to, for each product $i$:
\[
\begin{align*}
& x_i \leq d_i \\
& x_i \leq I_i \\
& x_i \in \mathbb{Z},\ x_i \geq 0
\end{align*}
\]
where $d_i$ is the Demand and $I_i$ is the Initial Inventory for product $i$.

Below are the parameters for each product, in source order:

| Index | Product Name           | Revenue ($A_i$) | Demand ($d_i$) | Initial Inventory ($I_i$) |
|-------|-----------------------|-----------------|----------------|--------------------------|
| 1     | Beauty - 25           | 25              | 240            | 1570                     |
| 2     | Beauty - 30           | 30              | 202            | 1330                     |
| 3     | Beauty - 300          | 300             | 216            | 1420                     |
| 4     | Beauty - 50           | 50              | 263            | 1700                     |
| 5     | Beauty - 500          | 500             | 256            | 1690                     |
| 6     | Clothing - 25         | 25              | 281            | 1840                     |
| 7     | Clothing - 30         | 30              | 261            | 1710                     |
| 8     | Clothing - 300        | 300             | 295            | 1930                     |
| 9     | Clothing - 50         | 50              | 290            | 1890                     |
| 10    | Clothing - 500        | 500             | 244            | 1570                     |
| 11    | Electronics - 25      | 25              | 273            | 1810                     |
| 12    | Electronics - 30      | 30              | 220            | 1410                     |
| 13    | Electronics - 300     | 300             | 286            | 1830                     |
| 14    | Electronics - 50      | 50              | 268            | 1750                     |
| 15    | Electronics - 500     | 500             | 262            | 1690                     |
| 16    | Home Goods - 25       | 25              | 255            | 1660                     |
| 17    | Home Goods - 30       | 30              | 218            | 1417                     |
| 18    | Home Goods - 50       | 50              | 278            | 1807                     |
| 19    | Home Goods - 300      | 300             | 195            | 1268                     |
| 20    | Home Goods - 500      | 500             | 248            | 1612                     |
| 21    | Sports - 25           | 25              | 269            | 1749                     |
| 22    | Sports - 30           | 30              | 227            | 1476                     |
| 23    | Sports - 50           | 50              | 285            | 1853                     |
| 24    | Sports - 300          | 300             | 208            | 1352                     |
| 25    | Sports - 500          | 500             | 259            | 1684                     |
| 26    | Furniture - 25        | 25              | 242            | 1573                     |
| 27    | Furniture - 30        | 30              | 213            | 1385                     |
| 28    | Furniture - 50        | 50              | 272            | 1768                     |
| 29    | Furniture - 300       | 300             | 224            | 1456                     |
| 30    | Furniture - 500       | 500             | 251            | 1632                     |
| 31    | Toys - 25             | 25              | 257            | 1671                     |
| 32    | Toys - 30             | 30              | 235            | 1528                     |
| 33    | Toys - 50             | 50              | 291            | 1892                     |
| 34    | Toys - 300            | 300             | 199            | 1294                     |
| 35    | Toys - 500            | 500             | 264            | 1716                     |

Decision variables:
\[
x_i = \text{number of units fulfilled for product } i, \quad x_i \in \mathbb{Z},\ x_i \geq 0
\]

Complete Model:
\[
\begin{align*}
\max \quad & \sum_{i=1}^{35} A_i x_i \\
\text{s.t.} \quad & x_i \leq d_i, \quad \forall i=1,\ldots,35 \\
& x_i \leq I_i, \quad \forall i=1,\ldots,35 \\
& x_i \in \mathbb{Z},\ x_i \geq 0, \quad \forall i=1,\ldots,35
\end{align*}
\]