##### Objective Function:

$\quad \max \sum_{i=1}^{10} v_i x_i$

where $x_i$ is the integer number of units of product $i$ to order, and $v_i$ is the benefit (Value) per unit of product $i$.

##### Constraints

###### 1. Capacity Constraint:

$\sum_{i=1}^{10} w_i x_i \leq 1035$

where $w_i$ is the weight per unit of product $i$.

###### 2. Variable Constraints:

$x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \{1,2,\ldots,10\}$

##### Retrieved Information

{
  "capacity": 1035,
  "products": [
    {
      "ProductName": "Spinach",
      "Weight": 282,
      "Value": 49
    },
    {
      "ProductName": "Shiitake Mushrooms",
      "Weight": 83,
      "Value": 30
    },
    {
      "ProductName": "Apples",
      "Weight": 251,
      "Value": 30
    },
    {
      "ProductName": "Carrots",
      "Weight": 257,
      "Value": 18
    },
    {
      "ProductName": "Basil",
      "Weight": 88,
      "Value": 54
    },
    {
      "ProductName": "Potatoes",
      "Weight": 52,
      "Value": 27
    },
    {
      "ProductName": "Green Beans",
      "Weight": 198,
      "Value": 91
    },
    {
      "ProductName": "Blueberries",
      "Weight": 203,
      "Value": 88
    },
    {
      "ProductName": "Oranges",
      "Weight": 87,
      "Value": 78
    },
    {
      "ProductName": "Watermelons",
      "Weight": 265,
      "Value": 22
    }
  ]
}

##### Parameter Vectors

Let the products be indexed in the order above ($i=1$ for Spinach, $i=2$ for Shiitake Mushrooms, ..., $i=10$ for Watermelons):

- $w = [282, 83, 251, 257, 88, 52, 198, 203, 87, 265]$
- $v = [49, 30, 30, 18, 54, 27, 91, 88, 78, 22]$

##### Decision Variables

- $x_i$: integer number of units of product $i$ to order daily, $x_i \geq 0$ for $i=1,\ldots,10$.

##### Complete Model

$\max \ 49x_1 + 30x_2 + 30x_3 + 18x_4 + 54x_5 + 27x_6 + 91x_7 + 88x_8 + 78x_9 + 22x_{10}$

subject to

$282x_1 + 83x_2 + 251x_3 + 257x_4 + 88x_5 + 52x_6 + 198x_7 + 203x_8 + 87x_9 + 265x_{10} \leq 1035$

$x_i \in \mathbb{Z}_{\geq 0}, \quad i=1,\ldots,10$