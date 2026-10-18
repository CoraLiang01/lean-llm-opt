##### Decision Variables

$x_{ij}\in \mathbb{Z}_{\geq0}$: number of units of product $j$ placed in section $i$.

##### Objective Function

$\displaystyle \max \sum_{i\in I}\sum_{j\in J}v_{j}\,x_{ij}$

##### Constraints

1. Capacity: $\displaystyle \sum_{j\in J}w_{j}\,x_{ij}\leq c_{i},\quad\forall i\in I$  
2. Integer non‑negativity: $x_{ij}\in \mathbb{Z}_{\geq0},\quad\forall i\in I,\;\forall j\in J$  

$ I=\{1,2,3,4,5,6,7,8\}$ and $J=\{1,2,3,4,5,6,7,8,9,10\}$.

###### Retrieved Information

{
  "sections": [
    {"SectionID":"1","Capacity":"100"},
    {"SectionID":"2","Capacity":"150"},
    {"SectionID":"3","Capacity":"120"},
    {"SectionID":"4","Capacity":"130"},
    {"SectionID":"5","Capacity":"90"},
    {"SectionID":"6","Capacity":"110"},
    {"SectionID":"7","Capacity":"160"},
    {"SectionID":"8","Capacity":"140"}
  ],
  "products": [
    {"ProductName":"1","Value":"10","Weight":"2"},
    {"ProductName":"2","Value":"15","Weight":"3"},
    {"ProductName":"3","Value":"8","Weight":"1"},
    {"ProductName":"4","Value":"12","Weight":"2"},
    {"ProductName":"5","Value":"20","Weight":"4"},
    {"ProductName":"6","Value":"25","Weight":"5"},
    {"ProductName":"7","Value":"5","Weight":"1"},
    {"ProductName":"8","Value":"30","Weight":"6"},
    {"ProductName":"9","Value":"18","Weight":"3"},
    {"ProductName":"10","Value":"22","Weight":"4"}
  ]
}