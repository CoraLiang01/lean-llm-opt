##### Sets

Let $I = \{1,2,3,4,5,6,7,8,9,10\}$ be the set of cabinets (CabinetID from capacity.csv).

Let $J =$ 
{
"Espresso Beans",
"Colombian Roast",
"Arabica Blend",
"French Roast",
"Italian Roast",
"House Blend",
"Sumatra Coffee",
"Mocha Java",
"Hazelnut Flavor",
"Caramel Blend",
"Vanilla Flavor",
"Cappuccino Mix",
"Pumpkin Spice",
"Decaf Roast",
"Organic Roast",
"Cold Brew",
"Peruvian Blend",
"Kenyan AA"
}
(the set of coffee products, ProductName from products.csv).

##### Parameters

Cabinet capacities (from capacity.csv):

- $C_1 = 400$
- $C_2 = 600$
- $C_3 = 500$
- $C_4 = 700$
- $C_5 = 450$
- $C_6 = 650$
- $C_7 = 550$
- $C_8 = 750$
- $C_9 = 480$
- $C_{10} = 520$

Product values and weights (from products.csv):

| $j$                | $v_j$ (Value) | $w_j$ (Weight) |
|--------------------|:-------------:|:--------------:|
| Espresso Beans     | 100           | 1.0            |
| Colombian Roast    | 150           | 1.5            |
| Arabica Blend      | 80            | 1.2            |
| French Roast       | 120           | 1.3            |
| Italian Roast      | 130           | 1.4            |
| House Blend        | 110           | 1.1            |
| Sumatra Coffee     | 160           | 1.8            |
| Mocha Java         | 90            | 1.2            |
| Hazelnut Flavor    | 95            | 1.0            |
| Caramel Blend      | 105           | 1.3            |
| Vanilla Flavor     | 85            | 1.2            |
| Cappuccino Mix     | 140           | 1.5            |
| Pumpkin Spice      | 75            | 1.1            |
| Decaf Roast        | 60            | 1.0            |
| Organic Roast      | 170           | 1.6            |
| Cold Brew          | 115           | 1.4            |
| Peruvian Blend     | 155           | 1.7            |
| Kenyan AA          | 125           | 1.3            |

##### Decision Variables

For each cabinet $i \in I$ and product $j \in J$:

$x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed in cabinet $i$.

##### Objective

$\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}$

##### Constraints

For each cabinet $i \in I$:

$\sum_{j \in J} w_j x_{ij} \leq C_i$

For all $i \in I$, $j \in J$:

$x_{ij} \in \mathbb{Z}_{\geq 0}$

##### Complete Numerical Formulation

$\max \Bigg[ \sum_{i=1}^{10} \Big($
$100\,x_{i,\text{Espresso Beans}} + 150\,x_{i,\text{Colombian Roast}} + 80\,x_{i,\text{Arabica Blend}} + 120\,x_{i,\text{French Roast}} + 130\,x_{i,\text{Italian Roast}} + 110\,x_{i,\text{House Blend}} + 160\,x_{i,\text{Sumatra Coffee}} + 90\,x_{i,\text{Mocha Java}} + 95\,x_{i,\text{Hazelnut Flavor}} + 105\,x_{i,\text{Caramel Blend}} + 85\,x_{i,\text{Vanilla Flavor}} + 140\,x_{i,\text{Cappuccino Mix}} + 75\,x_{i,\text{Pumpkin Spice}} + 60\,x_{i,\text{Decaf Roast}} + 170\,x_{i,\text{Organic Roast}} + 115\,x_{i,\text{Cold Brew}} + 155\,x_{i,\text{Peruvian Blend}} + 125\,x_{i,\text{Kenyan AA}} \Big) \Bigg]$

Subject to, for each $i=1,\ldots,10$:

$\begin{align*}
&1.0\,x_{i,\text{Espresso Beans}} + 1.5\,x_{i,\text{Colombian Roast}} + 1.2\,x_{i,\text{Arabica Blend}} + 1.3\,x_{i,\text{French Roast}} + 1.4\,x_{i,\text{Italian Roast}} + 1.1\,x_{i,\text{House Blend}} + 1.8\,x_{i,\text{Sumatra Coffee}} + 1.2\,x_{i,\text{Mocha Java}} \\
&+ 1.0\,x_{i,\text{Hazelnut Flavor}} + 1.3\,x_{i,\text{Caramel Blend}} + 1.2\,x_{i,\text{Vanilla Flavor}} + 1.5\,x_{i,\text{Cappuccino Mix}} + 1.1\,x_{i,\text{Pumpkin Spice}} + 1.0\,x_{i,\text{Decaf Roast}} + 1.6\,x_{i,\text{Organic Roast}} + 1.4\,x_{i,\text{Cold Brew}} \\
&+ 1.7\,x_{i,\text{Peruvian Blend}} + 1.3\,x_{i,\text{Kenyan AA}} \leq C_i
\end{align*}$

where $C_1=400$, $C_2=600$, $C_3=500$, $C_4=700$, $C_5=450$, $C_6=650$, $C_7=550$, $C_8=750$, $C_9=480$, $C_{10}=520$.

And for all $i=1,\ldots,10$, $j \in J$:

$x_{ij} \in \mathbb{Z}_{\geq 0}$

###### Retrieved Information

{
  "cabinets": [
    {"CabinetID":"1","Capacity":"400"},
    {"CabinetID":"2","Capacity":"600"},
    {"CabinetID":"3","Capacity":"500"},
    {"CabinetID":"4","Capacity":"700"},
    {"CabinetID":"5","Capacity":"450"},
    {"CabinetID":"6","Capacity":"650"},
    {"CabinetID":"7","Capacity":"550"},
    {"CabinetID":"8","Capacity":"750"},
    {"CabinetID":"9","Capacity":"480"},
    {"CabinetID":"10","Capacity":"520"}
  ],
  "products": [
    {"ProductName":"Espresso Beans","Value":"100","Weight":"1.0"},
    {"ProductName":"Colombian Roast","Value":"150","Weight":"1.5"},
    {"ProductName":"Arabica Blend","Value":"80","Weight":"1.2"},
    {"ProductName":"French Roast","Value":"120","Weight":"1.3"},
    {"ProductName":"Italian Roast","Value":"130","Weight":"1.4"},
    {"ProductName":"House Blend","Value":"110","Weight":"1.1"},
    {"ProductName":"Sumatra Coffee","Value":"160","Weight":"1.8"},
    {"ProductName":"Mocha Java","Value":"90","Weight":"1.2"},
    {"ProductName":"Hazelnut Flavor","Value":"95","Weight":"1.0"},
    {"ProductName":"Caramel Blend","Value":"105","Weight":"1.3"},
    {"ProductName":"Vanilla Flavor","Value":"85","Weight":"1.2"},
    {"ProductName":"Cappuccino Mix","Value":"140","Weight":"1.5"},
    {"ProductName":"Pumpkin Spice","Value":"75","Weight":"1.1"},
    {"ProductName":"Decaf Roast","Value":"60","Weight":"1.0"},
    {"ProductName":"Organic Roast","Value":"170","Weight":"1.6"},
    {"ProductName":"Cold Brew","Value":"115","Weight":"1.4"},
    {"ProductName":"Peruvian Blend","Value":"155","Weight":"1.7"},
    {"ProductName":"Kenyan AA","Value":"125","Weight":"1.3"}
  ]
}