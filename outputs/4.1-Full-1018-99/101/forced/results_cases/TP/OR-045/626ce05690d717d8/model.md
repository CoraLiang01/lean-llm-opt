##### Decision Variables

Let $x_i \in \mathbb{Z}_{\geq 0}$ denote the number of units of product $i$ to order daily, for each $i$ in the set of products below.

##### Parameters (from retrieved data)

Let the set of products $I$ be, in source order:
- Spinach
- Shiitake Mushrooms
- Apples
- Carrots
- Basil
- Potatoes
- Green Beans
- Blueberries
- Oranges
- Watermelons

Let $w_i$ be the weight per unit and $v_i$ the value per unit for each product $i$:

| Product               | $w_i$ (Weight) | $v_i$ (Value) |
|-----------------------|:--------------:|:-------------:|
| Spinach               | 282            | 49            |
| Shiitake Mushrooms    | 83             | 30            |
| Apples                | 251            | 30            |
| Carrots               | 257            | 18            |
| Basil                 | 88             | 54            |
| Potatoes              | 52             | 27            |
| Green Beans           | 198            | 91            |
| Blueberries           | 203            | 88            |
| Oranges               | 87             | 78            |
| Watermelons           | 265            | 22            |

Total inventory capacity: $C = 1035$

##### Mathematical Model

$\max \quad 49x_1 + 30x_2 + 30x_3 + 18x_4 + 54x_5 + 27x_6 + 91x_7 + 88x_8 + 78x_9 + 22x_{10}$

subject to

$282x_1 + 83x_2 + 251x_3 + 257x_4 + 88x_5 + 52x_6 + 198x_7 + 203x_8 + 87x_9 + 265x_{10} \leq 1035$

$x_i \in \mathbb{Z}_{\geq 0}$ for $i=1,\ldots,10$

where the correspondence between $x_i$ and product is as follows (in source order):

- $x_1$: Spinach
- $x_2$: Shiitake Mushrooms
- $x_3$: Apples
- $x_4$: Carrots
- $x_5$: Basil
- $x_6$: Potatoes
- $x_7$: Green Beans
- $x_8$: Blueberries
- $x_9$: Oranges
- $x_{10}$: Watermelons

###### Retrieved Information

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