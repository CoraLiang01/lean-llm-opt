##### Decision Variables

$x_i \geq 0$: number of units of product $i$ to order each day, for each $i \in P$ (continuous).

##### Parameters

Let $P$ be the set of products:
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

For each product $i \in P$:
- $w_i$: weight per unit of product $i$
- $v_i$: value (benefit) per unit of product $i$

Product data:

| Product Name         | $w_i$ (Weight) | $v_i$ (Value) |
|----------------------|:--------------:|:-------------:|
| Spinach              | 230            | 64            |
| Shiitake Mushrooms   | 637            | 75            |
| Apples               | 773            | 68            |
| Carrots              | 653            | 11            |
| Basil                | 755            | 91            |
| Potatoes             | 670            | 31            |
| Green Beans          | 505            | 90            |
| Blueberries          | 821            | 56            |
| Oranges              | 83             | 10            |
| Watermelons          | 249            | 24            |

Total stock capacity: $C = 875$

##### Objective Function

\[
\max \sum_{i \in P} v_i x_i
\]

##### Constraints

1. Stock capacity constraint:
   \[
   \sum_{i \in P} w_i x_i \leq C
   \]
2. Nonnegativity:
   \[
   x_i \geq 0 \quad \forall i \in P
   \]

##### Full Model

\[
\begin{align*}
\max\quad & 64x_{\text{Spinach}} + 75x_{\text{Shiitake Mushrooms}} + 68x_{\text{Apples}} + 11x_{\text{Carrots}} + 91x_{\text{Basil}} \\
& + 31x_{\text{Potatoes}} + 90x_{\text{Green Beans}} + 56x_{\text{Blueberries}} + 10x_{\text{Oranges}} + 24x_{\text{Watermelons}} \\
\text{s.t.}\quad & 230x_{\text{Spinach}} + 637x_{\text{Shiitake Mushrooms}} + 773x_{\text{Apples}} + 653x_{\text{Carrots}} + 755x_{\text{Basil}} \\
& + 670x_{\text{Potatoes}} + 505x_{\text{Green Beans}} + 821x_{\text{Blueberries}} + 83x_{\text{Oranges}} + 249x_{\text{Watermelons}} \leq 875 \\
& x_i \geq 0 \quad \forall i \in P
\end{align*}
\]

###### Retrieved Information

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