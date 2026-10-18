##### Decision Variables

$x_{ij}\in \mathbb{Z}_{\geq 0}$: number of units of product $j\in J$ placed on shelf $i\in I$.

##### Objective Function

$\max \sum_{i\in I}\sum_{j\in J} v_{j}\,x_{ij}$

##### Constraints

1. Shelf capacity: $\displaystyle \sum_{j\in J} w_{j}\,x_{ij}\leq C_{i},\quad \forall i\in I$.

2. Domains: $x_{ij}\in \mathbb{Z}_{\geq 0}$.

##### Sets

$I=\{1,2,3,4,5,6,7,8,9,10\}$,  
$J=\{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20\}$.

##### Parameters

Capacity vector $C$ (shelf capacities):

$C_{1}=500,\; C_{2}=700,\; C_{3}=600,\; C_{4}=800,\; C_{5}=550,\; C_{6}=900,\; C_{7}=650,\; C_{8}=750,\; C_{9}=820,\; C_{10}=570$.

Product value vector $v$:

$v_{1}=50,\; v_{2}=70,\; v_{3}=30,\; v_{4}=60,\; v_{5}=80,\; v_{6}=90,\; v_{7}=40,\; v_{8}=100,\; v_{9}=55,\; v_{10}=75,\; v_{11}=65,\; v_{12}=95,\; v_{13}=45,\; v_{14}=85,\; v_{15}=70,\; v_{16}=110,\; v_{17}=50,\; v_{18}=60,\; v_{19}=120,\; v_{20}=100$.

Product weight vector $w$:

$w_{1}=10,\; w_{2}=20,\; w_{3}=5,\; w_{4}=15,\; w_{5}=25,\; w_{6}=30,\; w_{7}=12,\; w_{8}=35,\; w_{9}=10,\; w_{10}=20,\; w_{11}=18,\; w_{12}=28,\; w_{13}=8,\; w_{14}=22,\; w_{15}=25,\; w_{16}=40,\; w_{17}=14,\; w_{18}=16,\; w_{19}=50,\; w_{20}=30$.

###### Retrieved Information

{
  "shelves": [
    {"ShelfID":"1","Capacity":"500"},
    {"ShelfID":"2","Capacity":"700"},
    {"ShelfID":"3","Capacity":"600"},
    {"ShelfID":"4","Capacity":"800"},
    {"ShelfID":"5","Capacity":"550"},
    {"ShelfID":"6","Capacity":"900"},
    {"ShelfID":"7","Capacity":"650"},
    {"ShelfID":"8","Capacity":"750"},
    {"ShelfID":"9","Capacity":"820"},
    {"ShelfID":"10","Capacity":"570"}
  ],
  "products": [
    {"ProductName":"1","Value":"50","Weight":"10"},
    {"ProductName":"2","Value":"70","Weight":"20"},
    {"ProductName":"3","Value":"30","Weight":"5"},
    {"ProductName":"4","Value":"60","Weight":"15"},
    {"ProductName":"5","Value":"80","Weight":"25"},
    {"ProductName":"6","Value":"90","Weight":"30"},
    {"ProductName":"7","Value":"40","Weight":"12"},
    {"ProductName":"8","Value":"100","Weight":"35"},
    {"ProductName":"9","Value":"55","Weight":"10"},
    {"ProductName":"10","Value":"75","Weight":"20"},
    {"ProductName":"11","Value":"65","Weight":"18"},
    {"ProductName":"12","Value":"95","Weight":"28"},
    {"ProductName":"13","Value":"45","Weight":"8"},
    {"ProductName":"14","Value":"85","Weight":"22"},
    {"ProductName":"15","Value":"70","Weight":"25"},
    {"ProductName":"16","Value":"110","Weight":"40"},
    {"ProductName":"17","Value":"50","Weight":"14"},
    {"ProductName":"18","Value":"60","Weight":"16"},
    {"ProductName":"19","Value":"120","Weight":"50"},
    {"ProductName":"20","Value":"100","Weight":"30"}
  ]
}