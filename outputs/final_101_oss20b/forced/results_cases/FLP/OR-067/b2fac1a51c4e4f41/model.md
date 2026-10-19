##### Decision Variables

$x_{ij}\in\{0,1\}$: manager $i\in I$ is assigned to project $j\in J$.

##### Objective Function

$\displaystyle \min \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}$

##### Constraints

1. Manager assignment: $\displaystyle \sum_{j\in J}x_{ij}=1,\quad \forall i\in I$  
2. Project assignment: $\displaystyle \sum_{i\in I}x_{ij}=1,\quad \forall j\in J$  
3. Domain: $x_{ij}\in\{0,1\}$.

##### Sets

$I=\{MA,MB,MC\}$, $J=\{P1,P2,P3\}$.

##### Cost Parameters

$c_{MA,P1}=3000,\;c_{MA,P2}=3200,\;c_{MA,P3}=3100$  
$c_{MB,P1}=2800,\;c_{MB,P2}=3300,\;c_{MB,P3}=2900$  
$c_{MC,P1}=2900,\;c_{MC,P2}=3100,\;c_{MC,P3}=3000$

###### Retrieved Information

{
  "managers": ["MA","MB","MC"],
  "projects": ["P1","P2","P3"],
  "cost_matrix": {
    "MA": {"P1":3000,"P2":3200,"P3":3100},
    "MB": {"P1":2800,"P2":3300,"P3":2900},
    "MC": {"P1":2900,"P2":3100,"P3":3000}
  }
}