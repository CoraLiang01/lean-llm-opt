##### Objective Function:

$\quad \max \sum_{i=1}^{10} v_i x_i$

where $x_i$ is the number of units of product $i$ to order, and $v_i$ is the value (benefit) per unit of product $i$.

##### Constraints

###### 1. Stock Capacity Constraint:

$\sum_{i=1}^{10} w_i x_i \leq 875$

where $w_i$ is the weight per unit of product $i$.

###### 2. Non-negativity and Integrality:

$x_i \geq 0$, integer, for all $i = 1, \ldots, 10$

##### Retrieved Information

{
  "capacity": 875,
  "products": [
    {
      "ProductName": "Spinach",
      "Weight": 230,
      "Value": 64
    },
    {
      "ProductName": "Shiitake Mushrooms",
      "Weight": 637,
      "Value": 75
    },
    {
      "ProductName": "Apples",
      "Weight": 773,
      "Value": 68
    },
    {
      "ProductName": "Carrots",
      "Weight": 653,
      "Value": 11
    },
    {
      "ProductName": "Basil",
      "Weight": 755,
      "Value": 91
    },
    {
      "ProductName": "Potatoes",
      "Weight": 670,
      "Value": 31
    },
    {
      "ProductName": "Green Beans",
      "Weight": 505,
      "Value": 90
    },
    {
      "ProductName": "Blueberries",
      "Weight": 821,
      "Value": 56
    },
    {
      "ProductName": "Oranges",
      "Weight": 83,
      "Value": 10
    },
    {
      "ProductName": "Watermelons",
      "Weight": 249,
      "Value": 24
    }
  ]
}

##### Parameter Vectors

Let the products be indexed in the order listed above ($i=1$ for Spinach, $i=2$ for Shiitake Mushrooms, ..., $i=10$ for Watermelons):

- $w = [230, 637, 773, 653, 755, 670, 505, 821, 83, 249]$
- $v = [64, 75, 68, 11, 91, 31, 90, 56, 10, 24]$
- Capacity $= 875$

##### Decision Variables

- $x_i$: integer, number of units of product $i$ to order each day, $i=1,\ldots,10$