##### Objective Function:

$\quad \min \sum_{i=1}^7 \sum_{j=1}^7 c_{ij} x_{ij}$

where $c_{ij}$ is the cost for Manager $i$ to complete Project $j$, and $x_{ij}$ is a binary variable equal to 1 if Manager $i$ is assigned to Project $j$, 0 otherwise.

##### Constraints

###### 1. Assignment Constraints:

Each manager is assigned to exactly one project:
$$
\sum_{j=1}^7 x_{ij} = 1 \quad \forall i \in \{1,2,3,4,5,6,7\}
$$

Each project is assigned to exactly one manager:
$$
\sum_{i=1}^7 x_{ij} = 1 \quad \forall j \in \{1,2,3,4,5,6,7\}
$$

###### 2. Variable Constraints:

$$
x_{ij} \in \{0,1\} \quad \forall i,j
$$

##### Retrieved Information

{
  "cost": {
    "Manager 1": {
      "Project 1 Cost": 2972,
      "Project 2 Cost": 2727,
      "Project 3 Cost": 2795,
      "Project 4 Cost": 2922,
      "Project 5 Cost": 1302,
      "Project 6 Cost": 2489,
      "Project 7 Cost": 1533
    },
    "Manager 2": {
      "Project 1 Cost": 1094,
      "Project 2 Cost": 2158,
      "Project 3 Cost": 2990,
      "Project 4 Cost": 1844,
      "Project 5 Cost": 2887,
      "Project 6 Cost": 2021,
      "Project 7 Cost": 2288
    },
    "Manager 3": {
      "Project 1 Cost": 2133,
      "Project 2 Cost": 1675,
      "Project 3 Cost": 2422,
      "Project 4 Cost": 2639,
      "Project 5 Cost": 1033,
      "Project 6 Cost": 2261,
      "Project 7 Cost": 1695
    },
    "Manager 4": {
      "Project 1 Cost": 1951,
      "Project 2 Cost": 2309,
      "Project 3 Cost": 2070,
      "Project 4 Cost": 2802,
      "Project 5 Cost": 2328,
      "Project 6 Cost": 1313,
      "Project 7 Cost": 2434
    },
    "Manager 5": {
      "Project 1 Cost": 1269,
      "Project 2 Cost": 2153,
      "Project 3 Cost": 1296,
      "Project 4 Cost": 2685,
      "Project 5 Cost": 2627,
      "Project 6 Cost": 1610,
      "Project 7 Cost": 1641
    },
    "Manager 6": {
      "Project 1 Cost": 1220,
      "Project 2 Cost": 1192,
      "Project 3 Cost": 2907,
      "Project 4 Cost": 2622,
      "Project 5 Cost": 2595,
      "Project 6 Cost": 1261,
      "Project 7 Cost": 2384
    },
    "Manager 7": {
      "Project 1 Cost": 1286,
      "Project 2 Cost": 1659,
      "Project 3 Cost": 1179,
      "Project 4 Cost": 1348,
      "Project 5 Cost": 1420,
      "Project 6 Cost": 2862,
      "Project 7 Cost": 1959
    }
  },
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
  ]
}