##### Decision Variables

$x_i\in\mathbb{Z}_{\geq0}$: number of units of product $i$ to be ordered each day, for each $i\in P$.

##### Objective Function

$\displaystyle \max\sum_{i\in P}v_i\,x_i$

##### Constraints

1. Storage capacity: $\displaystyle \sum_{i\in P}w_i\,x_i\leq 180$  
2. Non‑negativity and integrality: $\displaystyle x_i\in\mathbb{Z}_{\geq0},\quad\forall i\in P$

$P=\{\text{Baguette},\text{Croissant},\text{Sourdough},\text{Rye Bread},\text{Brioche},\text{Focaccia},\text{Ciabatta},\text{Pita},\text{Bagel},\text{English Muffin}\}$

###### Retrieved Information

{
  "capacity": 180,
  "products": [
    {"ProductName":"Baguette","Value":888,"Weight":4},
    {"ProductName":"Croissant","Value":134,"Weight":2},
    {"ProductName":"Sourdough","Value":129,"Weight":4},
    {"ProductName":"Rye Bread","Value":370,"Weight":3},
    {"ProductName":"Brioche","Value":921,"Weight":2},
    {"ProductName":"Focaccia","Value":765,"Weight":1},
    {"ProductName":"Ciabatta","Value":154,"Weight":2},
    {"ProductName":"Pita","Value":837,"Weight":1},
    {"ProductName":"Bagel","Value":584,"Weight":3},
    {"ProductName":"English Muffin","Value":365,"Weight":3}
  ]
}