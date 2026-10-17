##### Decision Variables

Let $x_i$ denote the fulfilled quantity for product $i$, where $i$ indexes the products/categories listed below.

##### Objective Function

$\max \sum_{i} \text{Revenue}_i \cdot x_i$

##### Constraints

For each product $i$:

$0 \leq x_i \leq \min(\text{Demand}_i, \text{Initial Inventory}_i)$

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

##### Model Summary

Let $i$ index the following products/categories:

- Beauty - 25, Beauty - 30, Beauty - 300, Beauty - 50, Beauty - 500
- Clothing - 25, Clothing - 30, Clothing - 300, Clothing - 50, Clothing - 500
- Electronics - 25, Electronics - 30, Electronics - 300, Electronics - 50, Electronics - 500
- Home Goods - 25, Home Goods - 30, Home Goods - 50, Home Goods - 300, Home Goods - 500
- Sports - 25, Sports - 30, Sports - 50, Sports - 300, Sports - 500
- Furniture - 25, Furniture - 30, Furniture - 50, Furniture - 300, Furniture - 500
- Toys - 25, Toys - 30, Toys - 50, Toys - 300, Toys - 500

For each $i$:

- $\text{Revenue}_i$ as listed above
- $\text{Demand}_i$ as listed above
- $\text{Initial Inventory}_i$ as listed above

Decision variable: $x_i$ (fulfilled quantity for product $i$)

Objective:

$\max \sum_{i} \text{Revenue}_i \cdot x_i$

Subject to:

$0 \leq x_i \leq \min(\text{Demand}_i, \text{Initial Inventory}_i)$ for all $i$