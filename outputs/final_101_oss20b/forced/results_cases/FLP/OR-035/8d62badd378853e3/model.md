##### Decision Variables

$x_i \in \mathbb{Z}_{\geq 0}$: number of units of bread type $i$ to be ordered each day, for all $i \in I$.

##### Objective Function

$\displaystyle \max \sum_{i\in I} p_i\,x_i$

##### Constraints

1. Storage capacity: $\displaystyle \sum_{i\in I} w_i\,x_i \;\leq\; C$.

2. Integer domain: $\displaystyle x_i \in \mathbb{Z}_{\geq 0},\quad \forall i\in I$.

##### Parameters

$I=\{\text{Baguette},\text{Croissant},\text{Sourdough},\text{Rye Bread},\text{Brioche},\text{Focaccia},\text{Ciabatta},\text{Pita},\text{Bagel},\text{English Muffin}\}$  

$C=180$  

$p=\begin{bmatrix}
888\\
134\\
129\\
370\\
921\\
765\\
154\\
837\\
584\\
365
\end{bmatrix}$  

$w=\begin{bmatrix}
4\\
2\\
4\\
3\\
2\\
1\\
2\\
1\\
3\\
3
\end{bmatrix}$  

###### Retrieved Information

{
  "capacity": 180,
  "products": [
    {"ProductName":"Baguette","Value":"888","Weight":"4"},
    {"ProductName":"Croissant","Value":"134","Weight":"2"},
    {"ProductName":"Sourdough","Value":"129","Weight":"4"},
    {"ProductName":"Rye Bread","Value":"370","Weight":"3"},
    {"ProductName":"Brioche","Value":"921","Weight":"2"},
    {"ProductName":"Focaccia","Value":"765","Weight":"1"},
    {"ProductName":"Ciabatta","Value":"154","Weight":"2"},
    {"ProductName":"Pita","Value":"837","Weight":"1"},
    {"ProductName":"Bagel","Value":"584","Weight":"3"},
    {"ProductName":"English Muffin","Value":"365","Weight":"3"}
  ]
}