Let $I$ be the set of product categories as listed below. For each $i\in I$:

- $r_i$: revenue per unit for product $i$
- $d_i$: demand for product $i$
- $s_i$: initial inventory for product $i$
- $x_i$: quantity fulfilled for product $i$ (decision variable, continuous, $x_i\geq0$)

Maximize total revenue:
$$
\max \sum_{i\in I} r_i x_i
$$

Subject to, for all $i\in I$:
$$
0 \leq x_i \leq \min\{d_i,\, s_i\}
$$

where the data is:

| Product Name           | $r_i$ | $d_i$ | $s_i$ |
|----------------------- |-------|-------|-------|
| Beauty - 25            | 25    | 240   | 1570  |
| Beauty - 30            | 30    | 202   | 1330  |
| Beauty - 300           | 300   | 216   | 1420  |
| Beauty - 50            | 50    | 263   | 1700  |
| Beauty - 500           | 500   | 256   | 1690  |
| Clothing - 25          | 25    | 281   | 1840  |
| Clothing - 30          | 30    | 261   | 1710  |
| Clothing - 300         | 300   | 295   | 1930  |
| Clothing - 50          | 50    | 290   | 1890  |
| Clothing - 500         | 500   | 244   | 1570  |
| Electronics - 25       | 25    | 273   | 1810  |
| Electronics - 30       | 30    | 220   | 1410  |
| Electronics - 300      | 300   | 286   | 1830  |
| Electronics - 50       | 50    | 268   | 1750  |
| Electronics - 500      | 500   | 262   | 1690  |
| Home Goods - 25        | 25    | 255   | 1660  |
| Home Goods - 30        | 30    | 218   | 1417  |
| Home Goods - 50        | 50    | 278   | 1807  |
| Home Goods - 300       | 300   | 195   | 1268  |
| Home Goods - 500       | 500   | 248   | 1612  |
| Sports - 25            | 25    | 269   | 1749  |
| Sports - 30            | 30    | 227   | 1476  |
| Sports - 50            | 50    | 285   | 1853  |
| Sports - 300           | 300   | 208   | 1352  |
| Sports - 500           | 500   | 259   | 1684  |
| Furniture - 25         | 25    | 242   | 1573  |
| Furniture - 30         | 30    | 213   | 1385  |
| Furniture - 50         | 50    | 272   | 1768  |
| Furniture - 300        | 300   | 224   | 1456  |
| Furniture - 500        | 500   | 251   | 1632  |
| Toys - 25              | 25    | 257   | 1671  |
| Toys - 30              | 30    | 235   | 1528  |
| Toys - 50              | 50    | 291   | 1892  |
| Toys - 300             | 300   | 199   | 1294  |
| Toys - 500             | 500   | 264   | 1716  |

Decision variables:
$$
x_i \in [0,\, \min\{d_i,\, s_i\}],\quad \forall i\in I
$$

Objective:
$$
\max \left(
25\,x_{\text{Beauty - 25}} + 30\,x_{\text{Beauty - 30}} + 300\,x_{\text{Beauty - 300}} + 50\,x_{\text{Beauty - 50}} + 500\,x_{\text{Beauty - 500}}
+ 25\,x_{\text{Clothing - 25}} + 30\,x_{\text{Clothing - 30}} + 300\,x_{\text{Clothing - 300}} + 50\,x_{\text{Clothing - 50}} + 500\,x_{\text{Clothing - 500}}
+ 25\,x_{\text{Electronics - 25}} + 30\,x_{\text{Electronics - 30}} + 300\,x_{\text{Electronics - 300}} + 50\,x_{\text{Electronics - 50}} + 500\,x_{\text{Electronics - 500}}
+ 25\,x_{\text{Home Goods - 25}} + 30\,x_{\text{Home Goods - 30}} + 50\,x_{\text{Home Goods - 50}} + 300\,x_{\text{Home Goods - 300}} + 500\,x_{\text{Home Goods - 500}}
+ 25\,x_{\text{Sports - 25}} + 30\,x_{\text{Sports - 30}} + 50\,x_{\text{Sports - 50}} + 300\,x_{\text{Sports - 300}} + 500\,x_{\text{Sports - 500}}
+ 25\,x_{\text{Furniture - 25}} + 30\,x_{\text{Furniture - 30}} + 50\,x_{\text{Furniture - 50}} + 300\,x_{\text{Furniture - 300}} + 500\,x_{\text{Furniture - 500}}
+ 25\,x_{\text{Toys - 25}} + 30\,x_{\text{Toys - 30}} + 50\,x_{\text{Toys - 50}} + 300\,x_{\text{Toys - 300}} + 500\,x_{\text{Toys - 500}}
\right)
$$

Subject to, for each product $i$ (in the order above):
$$
0 \leq x_i \leq \min\{\text{Demand}_i,\, \text{Initial Inventory}_i\}
$$

All $x_i$ are continuous and nonnegative.