##### Decision Variables

Let $x_i$ be the integer number of units of product $i$ to order daily, for each product $i$ in the set of products.

##### Parameters

- Let $N = 10$ (number of products).
- Let the set of products and their parameters be:

| $i$ | Product Name         | Weight per unit $w_i$ | Value per unit $v_i$ |
|-----|----------------------|-----------------------|----------------------|
| 1   | Spinach              | 282                   | 49                   |
| 2   | Shiitake Mushrooms   | 83                    | 30                   |
| 3   | Apples               | 251                   | 30                   |
| 4   | Carrots              | 257                   | 18                   |
| 5   | Basil                | 88                    | 54                   |
| 6   | Potatoes             | 52                    | 27                   |
| 7   | Green Beans          | 198                   | 91                   |
| 8   | Blueberries          | 203                   | 88                   |
| 9   | Oranges              | 87                    | 78                   |
| 10  | Watermelons          | 265                   | 22                   |

- Total inventory capacity: $C = 1035$

##### Objective Function

$\quad \max \sum_{i=1}^{10} v_i x_i$

##### Constraints

1. **Capacity Constraint:**

$\sum_{i=1}^{10} w_i x_i \leq 1035$

2. **Integrality Constraints:**

$x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,2,\ldots,10\}$

##### Retrieved Information

{
  "capacity": 1035,
  "products": [
    {"ProductName": "Spinach", "Weight": 282, "Value": 49},
    {"ProductName": "Shiitake Mushrooms", "Weight": 83, "Value": 30},
    {"ProductName": "Apples", "Weight": 251, "Value": 30},
    {"ProductName": "Carrots", "Weight": 257, "Value": 18},
    {"ProductName": "Basil", "Weight": 88, "Value": 54},
    {"ProductName": "Potatoes", "Weight": 52, "Value": 27},
    {"ProductName": "Green Beans", "Weight": 198, "Value": 91},
    {"ProductName": "Blueberries", "Weight": 203, "Value": 88},
    {"ProductName": "Oranges", "Weight": 87, "Value": 78},
    {"ProductName": "Watermelons", "Weight": 265, "Value": 22}
  ]
}