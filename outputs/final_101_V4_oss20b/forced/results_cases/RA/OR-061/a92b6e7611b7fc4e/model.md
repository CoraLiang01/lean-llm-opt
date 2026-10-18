##### Objective Function: 

$ \quad \quad \min \sum_{i \in \{S1,S2,S3,S4,S5\}} \sum_{j \in \{C1,C2,C3,C4,C5\}} t_{ij} \cdot x_{ij} + \sum_{i \in \{S1,S2,S3,S4,S5\}} f_{i} \cdot y_{i} $

##### Constraints

###### 1. Demand Satisfaction Constraints: 

$ \quad \quad \sum_{i \in \{S1,S2,S3,S4,S5\}} x_{ij} \geq d_{j}, \quad \forall j \in \{C1,C2,C3,C4,C5\} $

###### 2. Supplier Activation Constraints: 

$ \quad \quad x_{ij} \leq 187 \cdot y_{i}, \quad \forall i \in \{S1,S2,S3,S4,S5\}, \forall j \in \{C1,C2,C3,C4,C5\} $

###### 3. Binary Supplier Variables: 

$ \quad \quad y_{i} \in \{0,1\}, \quad \forall i \in \{S1,S2,S3,S4,S5\} $

###### 4. Integer Quantity Variables: 

$ \quad \quad x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \{S1,S2,S3,S4,S5\}, \forall j \in \{C1,C2,C3,C4,C5\} $

###### 5. Parameter Definitions: 

$ \quad \quad f_{i} = $ fixed cost of supplier $i$  
$ \quad \quad t_{ij} = $ transportation cost per unit from supplier $i$ to branch $j$  
$ \quad \quad d_{j} = $ demand of branch $j$  

###### 6. Parameter Values (Retrieved Data)

{
  "suppliers": [
    {"supplier_id": "S1", "fixed_cost": 97.65},
    {"supplier_id": "S2", "fixed_cost": 99.76},
    {"supplier_id": "S3", "fixed_cost": 100.76},
    {"supplier_id": "S4", "fixed_cost": 105.32},
    {"supplier_id": "S5", "fixed_cost": 98.88}
  ],
  "demands": [
    {"branch": "C1", "demand": 143},
    {"branch": "C2", "demand": 6},
    {"branch": "C3", "demand": 10},
    {"branch": "C4", "demand": 25},
    {"branch": "C5", "demand": 3}
  ],
  "transportation_costs": [
    {"supplier_id": "S1", "C1": 150.74, "C2": 0.02, "C3": 49.13, "C4": 2080.15, "C5": 426.4},
    {"supplier_id": "S2", "C1": 233.05, "C2": 97.73, "C3": 49.84, "C4": 1982.39, "C5": 23.96},
    {"supplier_id": "S3", "C1": 55.68, "C2": 935.61, "C3": 4.03, "C4": 73.09, "C5": 525.32},
    {"supplier_id": "S4", "C1": 1483.82, "C2": 1801.08, "C3": 112.16, "C4": 816.05, "C5": 107.01},
    {"supplier_id": "S5", "C1": 1119.47, "C2": 884.31, "C3": 0.08, "C4": 1544.95, "C5": 543.67}
  ]
}