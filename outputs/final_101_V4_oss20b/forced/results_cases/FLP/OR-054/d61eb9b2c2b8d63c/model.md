##### Decision Variables

$x_{ij} \geq 0$: number of units of product $j \in J$ placed on shelf $i \in I$.

##### Objective Function

$\max \sum_{i\in I}\sum_{j\in J} v_j x_{ij}$

##### Constraints

1. Shelf capacity: $\sum_{j\in J} w_j x_{ij}\leq C_i,\quad \forall i\in I$.

##### Sets and Parameters

$I=\{1,2,3,4,5,6,7,8,9,10\}$, $J=\{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20\}$.

$C=\{750,820,570,800,550,900,650,800,850,900\}$.

$v=\{55,75,65,60,80,90,40,100,55,75,110,50,60,120,70,110,50,60,120,100\}$.

$w=\{10,20,5,15,25,35,45,55,65,20,18,28,8,28,25,40,55,70,85,100\}$.

All parameters are given in the order of the corresponding indices in $I$ and $J$.