##### Objective Function:

$\quad \max \sum_{i=1}^{11} v_i x_i$

where $x_i$ is the number of units of product $i$ to order, and $v_i$ is the value (benefit) per unit of product $i$.

##### Constraints:

$\sum_{i=1}^{11} w_i x_i \leq 875$

$x_i \geq 0 \quad \forall i \in \{1,2,\ldots,11\}$

##### Parameters (retrieved):

- Capacity: $875$

- Products (with weights and values):

| Index ($i$) | Product Name         | $w_i$ (Weight) | $v_i$ (Value) |
|-------------|---------------------|----------------|---------------|
| 1           | Spinach             | 230            | 64            |
| 2           | Shiitake Mushrooms  | 637            | 75            |
| 3           | Apples              | 773            | 68            |
| 4           | Carrots             | 653            | 11            |
| 5           | Basil               | 755            | 91            |
| 6           | Potatoes            | 670            | 31            |
| 7           | Green Beans         | 505            | 90            |
| 8           | Blueberries         | 821            | 56            |
| 9           | Oranges             | 83             | 10            |
| 10          | Watermelons         | 249            | 24            |

##### Decision Variables:

$x_i$: Number of units of product $i$ to order each day, for $i = 1, \ldots, 11$.

##### Full Model:

$\max \left(64x_1 + 75x_2 + 68x_3 + 11x_4 + 91x_5 + 31x_6 + 90x_7 + 56x_8 + 10x_9 + 24x_{10} \right)$

subject to

$230x_1 + 637x_2 + 773x_3 + 653x_4 + 755x_5 + 670x_6 + 505x_7 + 821x_8 + 83x_9 + 249x_{10} \leq 875$

$x_i \geq 0 \quad \forall i \in \{1,2,\ldots,10\}$

##### Retrieved Information

{
  "capacity": 875,
  "products": [
    {"ProductName": "Spinach", "Weight": 230, "Value": 64},
    {"ProductName": "Shiitake Mushrooms", "Weight": 637, "Value": 75},
    {"ProductName": "Apples", "Weight": 773, "Value": 68},
    {"ProductName": "Carrots", "Weight": 653, "Value": 11},
    {"ProductName": "Basil", "Weight": 755, "Value": 91},
    {"ProductName": "Potatoes", "Weight": 670, "Value": 31},
    {"ProductName": "Green Beans", "Weight": 505, "Value": 90},
    {"ProductName": "Blueberries", "Weight": 821, "Value": 56},
    {"ProductName": "Oranges", "Weight": 83, "Value": 10},
    {"ProductName": "Watermelons", "Weight": 249, "Value": 24}
  ]
}