##### Decision Variables:

Let $x_i$ denote the fulfillment quantity for product $i$, where $i$ indexes the products/categories listed below.

##### Objective Function:

$\quad \max \sum_{i=1}^{35} r_i x_i$

where $r_i$ is the revenue per unit for product $i$.

##### Constraints:

For each product $i$:

$0 \leq x_i \leq \min\{\text{Initial Inventory}_i, \text{Demand}_i\}$

or, equivalently,

$0 \leq x_i \leq \text{Initial Inventory}_i$

$0 \leq x_i \leq \text{Demand}_i$

##### Variable Domains:

$x_i \geq 0$ and integer (if only whole units can be fulfilled; otherwise, $x_i \geq 0$ and continuous).

##### Retrieved Information

{
  "products": [
    {
      "Product Name": "Beauty - 25",
      "Revenue": 25,
      "Demand": 240,
      "Initial Inventory": 1570
    },
    {
      "Product Name": "Beauty - 30",
      "Revenue": 30,
      "Demand": 202,
      "Initial Inventory": 1330
    },
    {
      "Product Name": "Beauty - 300",
      "Revenue": 300,
      "Demand": 216,
      "Initial Inventory": 1420
    },
    {
      "Product Name": "Beauty - 50",
      "Revenue": 50,
      "Demand": 263,
      "Initial Inventory": 1700
    },
    {
      "Product Name": "Beauty - 500",
      "Revenue": 500,
      "Demand": 256,
      "Initial Inventory": 1690
    },
    {
      "Product Name": "Clothing - 25",
      "Revenue": 25,
      "Demand": 281,
      "Initial Inventory": 1840
    },
    {
      "Product Name": "Clothing - 30",
      "Revenue": 30,
      "Demand": 261,
      "Initial Inventory": 1710
    },
    {
      "Product Name": "Clothing - 300",
      "Revenue": 300,
      "Demand": 295,
      "Initial Inventory": 1930
    },
    {
      "Product Name": "Clothing - 50",
      "Revenue": 50,
      "Demand": 290,
      "Initial Inventory": 1890
    },
    {
      "Product Name": "Clothing - 500",
      "Revenue": 500,
      "Demand": 244,
      "Initial Inventory": 1570
    },
    {
      "Product Name": "Electronics - 25",
      "Revenue": 25,
      "Demand": 273,
      "Initial Inventory": 1810
    },
    {
      "Product Name": "Electronics - 30",
      "Revenue": 30,
      "Demand": 220,
      "Initial Inventory": 1410
    },
    {
      "Product Name": "Electronics - 300",
      "Revenue": 300,
      "Demand": 286,
      "Initial Inventory": 1830
    },
    {
      "Product Name": "Electronics - 50",
      "Revenue": 50,
      "Demand": 268,
      "Initial Inventory": 1750
    },
    {
      "Product Name": "Electronics - 500",
      "Revenue": 500,
      "Demand": 262,
      "Initial Inventory": 1690
    },
    {
      "Product Name": "Home Goods - 25",
      "Revenue": 25,
      "Demand": 255,
      "Initial Inventory": 1660
    },
    {
      "Product Name": "Home Goods - 30",
      "Revenue": 30,
      "Demand": 218,
      "Initial Inventory": 1417
    },
    {
      "Product Name": "Home Goods - 50",
      "Revenue": 50,
      "Demand": 278,
      "Initial Inventory": 1807
    },
    {
      "Product Name": "Home Goods - 300",
      "Revenue": 300,
      "Demand": 195,
      "Initial Inventory": 1268
    },
    {
      "Product Name": "Home Goods - 500",
      "Revenue": 500,
      "Demand": 248,
      "Initial Inventory": 1612
    },
    {
      "Product Name": "Sports - 25",
      "Revenue": 25,
      "Demand": 269,
      "Initial Inventory": 1749
    },
    {
      "Product Name": "Sports - 30",
      "Revenue": 30,
      "Demand": 227,
      "Initial Inventory": 1476
    },
    {
      "Product Name": "Sports - 50",
      "Revenue": 50,
      "Demand": 285,
      "Initial Inventory": 1853
    },
    {
      "Product Name": "Sports - 300",
      "Revenue": 300,
      "Demand": 208,
      "Initial Inventory": 1352
    },
    {
      "Product Name": "Sports - 500",
      "Revenue": 500,
      "Demand": 259,
      "Initial Inventory": 1684
    },
    {
      "Product Name": "Furniture - 25",
      "Revenue": 25,
      "Demand": 242,
      "Initial Inventory": 1573
    },
    {
      "Product Name": "Furniture - 30",
      "Revenue": 30,
      "Demand": 213,
      "Initial Inventory": 1385
    },
    {
      "Product Name": "Furniture - 50",
      "Revenue": 50,
      "Demand": 272,
      "Initial Inventory": 1768
    },
    {
      "Product Name": "Furniture - 300",
      "Revenue": 300,
      "Demand": 224,
      "Initial Inventory": 1456
    },
    {
      "Product Name": "Furniture - 500",
      "Revenue": 500,
      "Demand": 251,
      "Initial Inventory": 1632
    },
    {
      "Product Name": "Toys - 25",
      "Revenue": 25,
      "Demand": 257,
      "Initial Inventory": 1671
    },
    {
      "Product Name": "Toys - 30",
      "Revenue": 30,
      "Demand": 235,
      "Initial Inventory": 1528
    },
    {
      "Product Name": "Toys - 50",
      "Revenue": 50,
      "Demand": 291,
      "Initial Inventory": 1892
    },
    {
      "Product Name": "Toys - 300",
      "Revenue": 300,
      "Demand": 199,
      "Initial Inventory": 1294
    },
    {
      "Product Name": "Toys - 500",
      "Revenue": 500,
      "Demand": 264,
      "Initial Inventory": 1716
    }
  ]
}

##### Model Parameters Table

| $i$ | Product Name           | Revenue ($r_i$) | Demand ($d_i$) | Initial Inventory ($s_i$) |
|-----|------------------------|-----------------|---------------|--------------------------|
| 1   | Beauty - 25            | 25              | 240           | 1570                     |
| 2   | Beauty - 30            | 30              | 202           | 1330                     |
| 3   | Beauty - 300           | 300             | 216           | 1420                     |
| 4   | Beauty - 50            | 50              | 263           | 1700                     |
| 5   | Beauty - 500           | 500             | 256           | 1690                     |
| 6   | Clothing - 25          | 25              | 281           | 1840                     |
| 7   | Clothing - 30          | 30              | 261           | 1710                     |
| 8   | Clothing - 300         | 300             | 295           | 1930                     |
| 9   | Clothing - 50          | 50              | 290           | 1890                     |
| 10  | Clothing - 500         | 500             | 244           | 1570                     |
| 11  | Electronics - 25       | 25              | 273           | 1810                     |
| 12  | Electronics - 30       | 30              | 220           | 1410                     |
| 13  | Electronics - 300      | 300             | 286           | 1830                     |
| 14  | Electronics - 50       | 50              | 268           | 1750                     |
| 15  | Electronics - 500      | 500             | 262           | 1690                     |
| 16  | Home Goods - 25        | 25              | 255           | 1660                     |
| 17  | Home Goods - 30        | 30              | 218           | 1417                     |
| 18  | Home Goods - 50        | 50              | 278           | 1807                     |
| 19  | Home Goods - 300       | 300             | 195           | 1268                     |
| 20  | Home Goods - 500       | 500             | 248           | 1612                     |
| 21  | Sports - 25            | 25              | 269           | 1749                     |
| 22  | Sports - 30            | 30              | 227           | 1476                     |
| 23  | Sports - 50            | 50              | 285           | 1853                     |
| 24  | Sports - 300           | 300             | 208           | 1352                     |
| 25  | Sports - 500           | 500             | 259           | 1684                     |
| 26  | Furniture - 25         | 25              | 242           | 1573                     |
| 27  | Furniture - 30         | 30              | 213           | 1385                     |
| 28  | Furniture - 50         | 50              | 272           | 1768                     |
| 29  | Furniture - 300        | 300             | 224           | 1456                     |
| 30  | Furniture - 500        | 500             | 251           | 1632                     |
| 31  | Toys - 25              | 25              | 257           | 1671                     |
| 32  | Toys - 30              | 30              | 235           | 1528                     |
| 33  | Toys - 50              | 50              | 291           | 1892                     |
| 34  | Toys - 300             | 300             | 199           | 1294                     |
| 35  | Toys - 500             | 500             | 264           | 1716                     |

##### Full Mathematical Model

$\boxed{
\begin{align*}
\max \quad & \sum_{i=1}^{35} r_i x_i \\
\text{s.t.} \quad & 0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i = 1, \ldots, 35 \\
& x_i \geq 0 \quad \text{and integer (if required)} \\
\end{align*}
}$

where $r_i$, $d_i$, and $s_i$ are as listed above for each product $i$.