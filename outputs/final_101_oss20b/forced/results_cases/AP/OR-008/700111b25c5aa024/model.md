##### Objective Function: 

$\quad \quad \min \sum_{i=1}^{5} \sum_{j=1}^{6} c_{ij} x_{ij}$

##### Constraints

###### 1. Supply Constraints: 

$\sum_{j=1}^{6} x_{ij} \leq s_{i} \quad \forall i \in \{1,2,3,4,5\}$

###### 2. Demand Constraints: 

$\sum_{i=1}^{5} x_{ij} = d_{j} \quad \forall j \in \{1,2,3,4,5,6\}$

###### 3. Variable Constraints: 

$x_{ij} \geq 0 \quad \forall i,j$

###### Retrieved Information

{
  "cost": {
    "Supplier1": {
      "Customer1": 2,
      "Customer2": 3,
      "Customer3": 1,
      "Customer4": 2,
      "Customer5": 3,
      "Customer6": 2
    },
    "Supplier2": {
      "Customer1": 1,
      "Customer2": 2,
      "Customer3": 3,
      "Customer4": 2,
      "Customer5": 3,
      "Customer6": 2
    },
    "Supplier3": {
      "Customer1": 3,
      "Customer2": 1,
      "Customer3": 2,
      "Customer4": 3,
      "Customer5": 2,
      "Customer6": 3
    },
    "Supplier4": {
      "Customer1": 2,
      "Customer2": 3,
      "Customer3": 2,
      "Customer4": 1,
      "Customer5": 3,
      "Customer6": 4
    },
    "Supplier5": {
      "Customer1": 3,
      "Customer2": 2,
      "Customer3": 3,
      "Customer4": 3,
      "Customer5": 2,
      "Customer6": 3
    }
  },
  "demand": {
    "Customer1": 70,
    "Customer2": 80,
    "Customer3": 60,
    "Customer4": 90,
    "Customer5": 85,
    "Customer6": 95
  },
  "supply": {
    "Supplier1": 200,
    "Supplier2": 250,
    "Supplier3": 230,
    "Supplier4": 220,
    "Supplier5": 210
  }
}