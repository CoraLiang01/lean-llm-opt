##### Objective Function: 

$\quad \quad \max \sum_{i=1}^{10} v_{i} x_{i}$ 

##### Constraints

###### 1. Capacity Constraint: 

$\sum_{i=1}^{10} w_{i} x_{i} \leq C$

###### 2. Variable Constraints: 

$x_{i} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \{1,\dots,10\}$

###### Retrieved Information

{
  "capacity": 180,
  "products": [
    {"name":"Baguette","profit":888,"weight":4},
    {"name":"Croissant","profit":134,"weight":2},
    {"name":"Sourdough","profit":129,"weight":4},
    {"name":"Rye Bread","profit":370,"weight":3},
    {"name":"Brioche","profit":921,"weight":2},
    {"name":"Focaccia","profit":765,"weight":1},
    {"name":"Ciabatta","profit":154,"weight":2},
    {"name":"Pita","profit":837,"weight":1},
    {"name":"Bagel","profit":584,"weight":3},
    {"name":"English Muffin","profit":365,"weight":3}
  ],
  "variables": ["x1","x2","x3","x4","x5","x6","x7","x8","x9","x10"]
}