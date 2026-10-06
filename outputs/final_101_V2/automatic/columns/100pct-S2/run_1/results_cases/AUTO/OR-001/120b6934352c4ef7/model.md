##### Objective Function:

$\quad \min \sum_{i=1}^3 \sum_{j=1}^3 c_{ij} x_{ij}$

where $c_{ij}$ is the cost for manager $i$ to manage project $j$, and $x_{ij}$ is a binary variable indicating whether manager $i$ is assigned to project $j$.

##### Constraints

###### 1. Assignment Constraints:

$\sum_{j=1}^3 x_{ij} = 1 \quad \forall i \in \{1,2,3\}$

$\sum_{i=1}^3 x_{ij} = 1 \quad \forall j \in \{1,2,3\}$

###### 2. Variable Constraints:

$x_{ij} \in \{0,1\}, \quad \forall i,j$

##### Retrieved Information

{
  "cost": {
    "MA": {
      "P1": 3000,
      "P2": 3200,
      "P3": 3100
    },
    "MB": {
      "P1": 2800,
      "P2": 3300,
      "P3": 2900
    },
    "MC": {
      "P1": 2900,
      "P2": 3100,
      "P3": 3000
    }
  },
  "managers": [
    {
      "id": "MA",
      "annual_training_hours": 18,
      "manager_professional_association": "Construction",
      "manager_site_visit_count_2025_q4": 3,
      "operations_region": "East"
    },
    {
      "id": "MB",
      "annual_training_hours": 18,
      "manager_professional_association": "Civil",
      "manager_site_visit_count_2025_q4": 18,
      "operations_region": "East"
    },
    {
      "id": "MC",
      "annual_training_hours": 12,
      "manager_professional_association": "Civil",
      "manager_site_visit_count_2025_q4": 18,
      "operations_region": "East"
    }
  ],
  "projects": [
    "P1",
    "P2",
    "P3"
  ]
}