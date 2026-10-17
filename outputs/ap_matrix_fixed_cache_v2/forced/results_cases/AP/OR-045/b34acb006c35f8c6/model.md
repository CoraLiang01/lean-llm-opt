##### Decision Variables

Let $x_i$ be the integer number of units of product $i$ to order daily, for each product $i$ in the set:
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

##### Parameters

- $v_i$: Benefit per unit of product $i$
- $w_i$: Weight per unit of product $i$
- $C$: Total inventory capacity

From the retrieved data:

| Product Name         | $w_i$ (Weight) | $v_i$ (Value) |
|----------------------|:--------------:|:-------------:|
| Spinach              | 282            | 49            |
| Shiitake Mushrooms   | 83             | 30            |
| Apples               | 251            | 30            |
| Carrots              | 257            | 18            |
| Basil                | 88             | 54            |
| Potatoes             | 52             | 27            |
| Green Beans          | 198            | 91            |
| Blueberries          | 203            | 88            |
| Oranges              | 87             | 78            |
| Watermelons          | 265            | 22            |

Total capacity: $C = 1035$

##### Objective Function

$\max \left(49x_{\text{Spinach}} + 30x_{\text{Shiitake Mushrooms}} + 30x_{\text{Apples}} + 18x_{\text{Carrots}} + 54x_{\text{Basil}} + 27x_{\text{Potatoes}} + 91x_{\text{Green Beans}} + 88x_{\text{Blueberries}} + 78x_{\text{Oranges}} + 22x_{\text{Watermelons}}\right)$

##### Constraints

$\quad 282x_{\text{Spinach}} + 83x_{\text{Shiitake Mushrooms}} + 251x_{\text{Apples}} + 257x_{\text{Carrots}} + 88x_{\text{Basil}} + 52x_{\text{Potatoes}} + 198x_{\text{Green Beans}} + 203x_{\text{Blueberries}} + 87x_{\text{Oranges}} + 265x_{\text{Watermelons}} \leq 1035$

$x_i \in \mathbb{Z}_{\geq 0}$ for all products $i$

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