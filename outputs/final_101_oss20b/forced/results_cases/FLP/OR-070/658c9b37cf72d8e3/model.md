##### Decision Variables

$x_{ij}\in\{0,1\}$: assignment of manager $i\in I$ to project $j\in J$ (binary).  

##### Objective Function

$\min \sum_{i\in I}\sum_{j\in J} c_{ij}x_{ij}$  

##### Constraints  

1. Each project is assigned to exactly one manager:  
   $\sum_{i\in I}x_{ij}=1,\quad \forall j\in J$  

2. Each manager is assigned to at most one project:  
   $\sum_{j\in J}x_{ij}\leq 1,\quad \forall i\in I$  

3. Domains: $x_{ij}\in\{0,1\}$  

$I=\{\text{Manager 1},\text{Manager 2},\text{Manager 3},\text{Manager 4},\text{Manager 5},\text{Manager 6},\text{Manager 7}\}$  

$J=\{\text{Project 1},\text{Project 2},\text{Project 3},\text{Project 4},\text{Project 5},\text{Project 6},\text{Project 7}\}$  

###### Retrieved Information

{
  "managers": [
    "Manager 1",
    "Manager 2",
    "Manager 3",
    "Manager 4",
    "Manager 5",
    "Manager 6",
    "Manager 7"
  ],
  "projects": [
    "Project 1",
    "Project 2",
    "Project 3",
    "Project 4",
    "Project 5",
    "Project 6",
    "Project 7"
  ],
  "cost": [
    [2972, 2727, 2795, 2922, 1302, 2489, 1533],
    [1094, 2158, 2990, 1844, 2887, 2021, 2288],
    [2133, 1675, 2422, 2639, 1033, 2261, 1695],
    [1951, 2309, 2070, 2802, 2328, 1313, 2434],
    [1269, 2153, 1296, 2685, 2627, 1610, 1641],
    [1220, 1192, 2907, 2622, 2595, 1261, 2384],
    [1286, 1659, 1179, 1348, 1420, 2862, 1959]
  ]
}