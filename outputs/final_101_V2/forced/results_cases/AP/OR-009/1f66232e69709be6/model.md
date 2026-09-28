##### Objective Function:

$\quad \min \sum_{i=1}^4 \sum_{j=1}^4 c_{ij} \, x_{ij}$

where $x_{ij}$ is the number of units shipped from supply node $i$ to customer $j$, and $c_{ij}$ is the transportation cost per unit from supply node $i$ to customer $j$.

##### Constraints

###### 1. Demand Satisfaction (for each customer $j$):

$\sum_{i=1}^4 x_{ij} = d_j \quad \forall j \in \{1,2,3,4\}$

###### 2. Supply Capacity (for each supply node $i$):

$\sum_{j=1}^4 x_{ij} \leq s_i \quad \forall i \in \{1,2,3,4\}$

###### 3. Non-negativity:

$x_{ij} \geq 0 \quad \forall i,j$

##### Retrieved Information

{
  "customers": [
    {"id": "C1", "demand": 94},
    {"id": "C2", "demand": 39},
    {"id": "C3", "demand": 65},
    {"id": "C4", "demand": 435}
  ],
  "suppliers": [
    {"id": "S1", "capacity": 2531},
    {"id": "S2", "capacity": 20},
    {"id": "S3", "capacity": 210},
    {"id": "S4", "capacity": 241}
  ],
  "transportation_cost": {
    "S1": {"C1": 543.756480860856, "C2": 23.685276141764653, "C3": 23.676386730773032, "C4": 447.75143678673766},
    "S2": {"C1": 883.9151090405642, "C2": 0.04977684765576961, "C3": 0.0350986687216299, "C4": 44.45588531711622},
    "S3": {"C1": 537.3456896658107, "C2": 23.769274659075112, "C3": 498.95659249465467, "C4": 440.60737890439776},
    "S4": {"C1": 1791.493192397229, "C2": 68.21633865655126, "C3": 1432.4837339656747, "C4": 1527.7635425462734}
  }
}

##### Variable Definitions

$x_{ij}$: Number of beverage units shipped from supply node $i$ (where $i \in \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$) to customer $j$ (where $j \in \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$).

##### Full Model (with explicit indices):

$\min \Bigg[ $
$543.756480860856\,x_{S1,C1} + 23.685276141764653\,x_{S1,C2} + 23.676386730773032\,x_{S1,C3} + 447.75143678673766\,x_{S1,C4} +$
$883.9151090405642\,x_{S2,C1} + 0.04977684765576961\,x_{S2,C2} + 0.0350986687216299\,x_{S2,C3} + 44.45588531711622\,x_{S2,C4} +$
$537.3456896658107\,x_{S3,C1} + 23.769274659075112\,x_{S3,C2} + 498.95659249465467\,x_{S3,C3} + 440.60737890439776\,x_{S3,C4} +$
$1791.493192397229\,x_{S4,C1} + 68.21633865655126\,x_{S4,C2} + 1432.4837339656747\,x_{S4,C3} + 1527.7635425462734\,x_{S4,C4} $
$\Bigg]$

Subject to:

$\quad x_{S1,C1} + x_{S2,C1} + x_{S3,C1} + x_{S4,C1} = 94$

$\quad x_{S1,C2} + x_{S2,C2} + x_{S3,C2} + x_{S4,C2} = 39$

$\quad x_{S1,C3} + x_{S2,C3} + x_{S3,C3} + x_{S4,C3} = 65$

$\quad x_{S1,C4} + x_{S2,C4} + x_{S3,C4} + x_{S4,C4} = 435$

$\quad x_{S1,C1} + x_{S1,C2} + x_{S1,C3} + x_{S1,C4} \leq 2531$

$\quad x_{S2,C1} + x_{S2,C2} + x_{S2,C3} + x_{S2,C4} \leq 20$

$\quad x_{S3,C1} + x_{S3,C2} + x_{S3,C3} + x_{S3,C4} \leq 210$

$\quad x_{S4,C1} + x_{S4,C2} + x_{S4,C3} + x_{S4,C4} \leq 241$

$\quad x_{ij} \geq 0 \quad \forall i \in \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\},\ j \in \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$