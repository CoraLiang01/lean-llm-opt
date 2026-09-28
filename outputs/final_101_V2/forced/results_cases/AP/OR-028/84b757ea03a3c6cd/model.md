##### Decision Variables:

Let $x_i$ denote the number of units of product $i$ to be fulfilled, for each product $i$ in the set of products.

##### Objective Function:

$\quad \max \sum_{i} r_i x_i$

where $r_i$ is the revenue per unit for product $i$.

##### Constraints:

1. Inventory and Demand Constraints:

$\quad 0 \leq x_i \leq \min\{\text{Initial Inventory}_i, \text{Demand}_i\} \quad \forall i$

or, equivalently, for each product $i$:

$\quad 0 \leq x_i \leq \min\{\text{Initial Inventory}_i, \text{Demand}_i\}$

2. Variable Constraints:

$\quad x_i \in \mathbb{Z}_{\geq 0} \quad \forall i$

##### Retrieved Information

{
  "products": [
    {
      "Product Name": "sku_I27",
      "Revenue": 238,
      "Demand": 6,
      "Initial Inventory": 30
    },
    {
      "Product Name": "sku_I499",
      "Revenue": 287,
      "Demand": 4,
      "Initial Inventory": 20
    },
    {
      "Product Name": "sku_I719",
      "Revenue": 268,
      "Demand": 16,
      "Initial Inventory": 80
    },
    {
      "Product Name": "sku_T18",
      "Revenue": 318,
      "Demand": 14,
      "Initial Inventory": 70
    },
    {
      "Product Name": "sku_T29",
      "Revenue": 207,
      "Demand": 4,
      "Initial Inventory": 20
    },
    {
      "Product Name": "sku_T39",
      "Revenue": 258,
      "Demand": 32,
      "Initial Inventory": 160
    },
    {
      "Product Name": "sku_T499",
      "Revenue": 249,
      "Demand": 8,
      "Initial Inventory": 40
    },
    {
      "Product Name": "sku_T9",
      "Revenue": 227,
      "Demand": 2,
      "Initial Inventory": 10
    },
    {
      "Product Name": "sku_3081",
      "Revenue": 198,
      "Demand": 10,
      "Initial Inventory": 50
    },
    {
      "Product Name": "sku_339",
      "Revenue": 254,
      "Demand": 8,
      "Initial Inventory": 40
    },
    {
      "Product Name": "sku_3799",
      "Revenue": 246,
      "Demand": 18,
      "Initial Inventory": 90
    },
    {
      "Product Name": "sku_439",
      "Revenue": 258,
      "Demand": 2,
      "Initial Inventory": 10
    },
    {
      "Product Name": "sku_539",
      "Revenue": 268,
      "Demand": 4,
      "Initial Inventory": 20
    },
    {
      "Product Name": "sku_61399",
      "Revenue": 278,
      "Demand": 8,
      "Initial Inventory": 40
    },
    {
      "Product Name": "sku_628",
      "Revenue": 268,
      "Demand": 2,
      "Initial Inventory": 10
    },
    {
      "Product Name": "sku_708",
      "Revenue": 298,
      "Demand": 198,
      "Initial Inventory": 990
    },
    {
      "Product Name": "sku_77",
      "Revenue": 258,
      "Demand": 32,
      "Initial Inventory": 160
    },
    {
      "Product Name": "sku_79",
      "Revenue": 315,
      "Demand": 18,
      "Initial Inventory": 90
    },
    {
      "Product Name": "sku_799",
      "Revenue": 264,
      "Demand": 570,
      "Initial Inventory": 2870
    },
    {
      "Product Name": "sku_8499",
      "Revenue": 238,
      "Demand": 6,
      "Initial Inventory": 30
    },
    {
      "Product Name": "sku_89",
      "Revenue": 258,
      "Demand": 26,
      "Initial Inventory": 130
    },
    {
      "Product Name": "sku_897",
      "Revenue": 268,
      "Demand": 6,
      "Initial Inventory": 30
    },
    {
      "Product Name": "sku_9699",
      "Revenue": 288,
      "Demand": 33,
      "Initial Inventory": 170
    },
    {
      "Product Name": "sku_bobo",
      "Revenue": 228,
      "Demand": 33,
      "Initial Inventory": 170
    }
  ]
}

##### Explicit Model (with all parameters):

Let $I$ be the set of products:

$I = \{$sku_I27, sku_I499, sku_I719, sku_T18, sku_T29, sku_T39, sku_T499, sku_T9, sku_3081, sku_339, sku_3799, sku_439, sku_539, sku_61399, sku_628, sku_708, sku_77, sku_79, sku_799, sku_8499, sku_89, sku_897, sku_9699, sku_bobo$\}$

For each $i \in I$:

- $r_i$ = Revenue per unit (see table above)
- $d_i$ = Demand (see table above)
- $s_i$ = Initial Inventory (see table above)

$\max \sum_{i \in I} r_i x_i$

subject to

$0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in I$

$x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I$