##### Objective Function: 

$\quad \quad \max \sum_i p_i \cdot x_i$

##### Constraints

###### 1. Capacity Constraint: 

$\quad \quad \sum_i a_i x_i \leq C$

###### 2. Non-negativity and Integrality Constraints: 

$\quad \quad  x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i$

Where $p_i$ is the Value of product $i$, $a_i$ is the Weight of product $i$, and $C$ is the overall stock capacity.

###### Retrieved Information

{
  "items": [
    "Spinach",
    "Shiitake Mushrooms",
    "Apples",
    "Carrots",
    "Basil",
    "Potatoes",
    "Green Beans",
    "Blueberries",
    "Oranges",
    "Watermelons"
  ],
  "profit": {
    "Spinach": 64,
    "Shiitake Mushrooms": 75,
    "Apples": 68,
    "Carrots": 11,
    "Basil": 91,
    "Potatoes": 31,
    "Green Beans": 90,
    "Blueberries": 56,
    "Oranges": 10,
    "Watermelons": 24
  },
  "weight": {
    "Spinach": 230,
    "Shiitake Mushrooms": 637,
    "Apples": 773,
    "Carrots": 653,
    "Basil": 755,
    "Potatoes": 670,
    "Green Beans": 505,
    "Blueberries": 821,
    "Oranges": 83,
    "Watermelons": 249
  },
  "capacity": 875
}