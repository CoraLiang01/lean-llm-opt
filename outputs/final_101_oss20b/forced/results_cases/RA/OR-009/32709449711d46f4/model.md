##### Objective Function: 

$\quad \quad \min \sum_{i} \sum_{j} c_{ij} \cdot x_{ij}$

##### Constraints

###### 1. Supply Capacity Constraints: 

$\quad \quad \sum_{j} x_{ij} \leq s_{i}, \quad \forall i$

###### 2. Demand Satisfaction Constraints: 

$\quad \quad \sum_{i} x_{ij} \geq d_{j}, \quad \forall j$

###### 3. Non-negativity and Integrality Constraints: 

$\quad \quad  x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i,j $

Where $i$ indexes the production plants $S1,S2,S3,S4$, $j$ indexes the retail customers $C1,C2,C3,C4$, $c_{ij}$ is the transportation cost per unit from plant $i$ to customer $j$, $s_{i}$ is the supply capacity of plant $i$, and $d_{j}$ is the demand of customer $j$.

###### Retrieved Information

{
  "plants": [
    "S1",
    "S2",
    "S3",
    "S4"
  ],
  "supply_capacity": {
    "S1": 2531,
    "S2": 20,
    "S3": 210,
    "S4": 241
  },
  "customers": [
    "C1",
    "C2",
    "C3",
    "C4"
  ],
  "demand": {
    "C1": 94,
    "C2": 39,
    "C3": 65,
    "C4": 435
  },
  "transportation_costs": {
    "S1": {
      "C1": 543.756480860856,
      "C2": 23.685276141764653,
      "C3": 23.676386730773032,
      "C4": 447.75143678673766
    },
    "S2": {
      "C1": 883.9151090405642,
      "C2": 0.04977684765576961,
      "C3": 0.0350986687216299,
      "C4": 44.45588531711622
    },
    "S3": {
      "C1": 537.3456896658107,
      "C2": 23.769274659075112,
      "C3": 498.95659249465467,
      "C4": 440.60737890439776
    },
    "S4": {
      "C1": 1791.493192397229,
      "C2": 68.21633865655126,
      "C3": 1432.4837339656747,
      "C4": 1527.7635425462734
    }
  }
}