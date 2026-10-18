##### Objective Function:

$\quad \max \sum_{i=1}^8 \sum_{j=1}^{10} v_j \, x_{ij}$

where $x_{ij}$ is the number of units of product $j$ to be placed in section $i$, and $v_j$ is the value (price) of product $j$.

##### Constraints:

###### 1. Section Capacity Constraints:

$\sum_{j=1}^{10} w_j \, x_{ij} \leq C_i \quad \forall i \in \{1,2,3,4,5,6,7,8\}$

where $w_j$ is the shelf space requirement (weight) of product $j$, and $C_i$ is the capacity of section $i$.

###### 2. Variable Constraints:

$x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,2,3,4,5,6,7,8\}, \; j \in \{1,2,3,4,5,6,7,8,9,10\}$

##### Retrieved Information

{
  "sections": [
    {"SectionID": "1", "Capacity": 100},
    {"SectionID": "2", "Capacity": 150},
    {"SectionID": "3", "Capacity": 120},
    {"SectionID": "4", "Capacity": 130},
    {"SectionID": "5", "Capacity": 90},
    {"SectionID": "6", "Capacity": 110},
    {"SectionID": "7", "Capacity": 160},
    {"SectionID": "8", "Capacity": 140}
  ],
  "products": [
    {"ProductName": "1", "Value": 10, "Weight": 2},
    {"ProductName": "2", "Value": 15, "Weight": 3},
    {"ProductName": "3", "Value": 8, "Weight": 1},
    {"ProductName": "4", "Value": 12, "Weight": 2},
    {"ProductName": "5", "Value": 20, "Weight": 4},
    {"ProductName": "6", "Value": 25, "Weight": 5},
    {"ProductName": "7", "Value": 5, "Weight": 1},
    {"ProductName": "8", "Value": 30, "Weight": 6},
    {"ProductName": "9", "Value": 18, "Weight": 3},
    {"ProductName": "10", "Value": 22, "Weight": 4}
  ]
}

##### Parameter Vectors and Matrices

- Section capacities: $C = [100, 150, 120, 130, 90, 110, 160, 140]$
- Product values: $v = [10, 15, 8, 12, 20, 25, 5, 30, 18, 22]$
- Product weights: $w = [2, 3, 1, 2, 4, 5, 1, 6, 3, 4]$
- Decision variables: $x_{ij}$ for $i = 1,\ldots,8$, $j = 1,\ldots,10$