##### Decision Variables

$x_{ij}\in\{0,1\}$: worker $i\in I$ is assigned to task $j\in J$.

##### Objective Function

$\min \sum_{i\in I}\sum_{j\in J} c_{ij}x_{ij}$

##### Constraints

1. Task assignment: $\sum_{i\in I}x_{ij}=1,\quad \forall j\in J$  
2. Worker capacity: $\sum_{j\in J}x_{ij}\leq 1,\quad \forall i\in I$  
3. Domains: $x_{ij}\in\{0,1\}$.

$I=\{1,2,3,4,5,6,7,8,9,10,11,12\}$, $J=\{A,B,C,D,E,F,G,H,I,J\}$.

###### Retrieved Information

{
  "workers": [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12"
  ],
  "tasks": [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J"
  ],
  "time_matrix": [
    [9,4,3,7,6,5,6,3,7,5],
    [4,6,5,6,4,5,3,8,7,6],
    [5,4,7,5,6,6,5,8,6,9],
    [7,5,2,3,7,8,5,6,8,5],
    [10,6,7,4,5,4,4,5,9,7],
    [6,7,6,3,9,5,7,4,3,4],
    [8,8,5,9,5,7,5,9,5,3],
    [7,4,8,8,6,7,5,7,7,7],
    [5,6,8,7,7,8,7,8,4,5],
    [8,7,9,5,8,5,9,9,3,4],
    [9,8,10,8,5,4,7,6,8,7],
    [8,5,6,9,4,7,8,4,7,9]
  ]
}