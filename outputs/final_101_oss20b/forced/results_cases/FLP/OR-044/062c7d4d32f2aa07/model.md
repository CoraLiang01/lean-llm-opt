##### Decision Variables

$x_{ij}\in \mathbb{Z}_{\geq 0}$: number of units of product $j\in J$ placed in section $i\in I$.

##### Objective Function

$\displaystyle \max \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij}$

##### Constraints

1. Section capacity: $\displaystyle \sum_{j\in J} w_j\,x_{ij}\leq C_i,\quad \forall i\in I$  
2. Non‑negativity and integrality: $\displaystyle x_{ij}\in \mathbb{Z}_{\geq 0},\quad \forall i\in I,\;j\in J$

##### Sets and Parameters

$I=\{1,2,3,4,5,6,7,8\}$,  
$J=\{1,2,3,4,5,6,7,8,9,10\}$  

$C=\bigl[100,\;150,\;120,\;130,\;90,\;110,\;160,\;140\bigr]$  

$v=\bigl[10,\;15,\;8,\;12,\;20,\;25,\;5,\;30,\;18,\;22\bigr]$  

$w=\bigl[2,\;3,\;1,\;2,\;4,\;5,\;1,\;6,\;3,\;4\bigr]$

###### Retrieved Information

{
  "capacity": {
    "SectionID": [1,2,3,4,5,6,7,8],
    "Capacity": [100,150,120,130,90,110,160,140]
  },
  "products": {
    "ProductName": [1,2,3,4,5,6,7,8,9,10],
    "Value": [10,15,8,12,20,25,5,30,18,22],
    "Weight": [2,3,1,2,4,5,1,6,3,4]
  }
}