##### Objective Function: 

$\quad \quad \min \sum_{i} \sum_{j} c_{ij} \cdot x_{ij}$

##### Constraints

###### 1. Manager Assignment Constraints: 

$\quad \quad \sum_{j} x_{ij} = 1, \quad \forall i$

###### 2. Project Assignment Constraints: 

$\quad \quad \sum_{i} x_{ij} = 1, \quad \forall j$

###### 3. Binary Constraints: 

$\quad \quad x_{ij} \in \{0,1\}, \quad \forall i,j$

###### Retrieved Information

{
  "managers": ["MA","MB","MC"],
  "projects": ["P1","P2","P3"],
  "costs": {
    "MA": {"P1":3000,"P2":3200,"P3":3100},
    "MB": {"P1":2800,"P2":3300,"P3":2900},
    "MC": {"P1":2900,"P2":3100,"P3":3000}
  }
}