##### Objective Function: 

$\quad \quad \max \sum_{i=1}^{2} r_{i} x_{i}$

##### Constraints

###### 1. Demand Constraints: 

$\quad \quad x_{i} \leq d_{i} \quad \forall i \in \{1,2\}$

###### 2. Inventory Constraints: 

$\quad \quad x_{i} \leq I_{i} \quad \forall i \in \{1,2\}$

###### 3. Non‑negativity Constraints: 

$\quad \quad x_{i} \geq 0 \quad \forall i \in \{1,2\}$

###### 4. Integer Constraints: 

$\quad \quad x_{i} \in \mathbb{Z}_{+} \quad \forall i \in \{1,2\}$

###### 5. Variable Definitions: 

$\quad \quad x_{1} = \text{units of “27in 4K Gaming Monitor” to fulfill}$  

$\quad \quad x_{2} = \text{units of “27in FHD Monitor” to fulfill}$

###### 6. Parameter Values: 

$\quad \quad r_{1} = 389.99,\; d_{1} = 12474,\; I_{1} = 62440$  

$\quad \quad r_{2} = 149.99,\; d_{2} = 15057,\; I_{2} = 75500$

###### 7. Retrieved Information

{
  "products": [
    "27in 4K Gaming Monitor",
    "27in FHD Monitor"
  ],
  "revenue_per_unit": {
    "27in 4K Gaming Monitor": 389.99,
    "27in FHD Monitor": 149.99
  },
  "demand": {
    "27in 4K Gaming Monitor": 12474,
    "27in FHD Monitor": 15057
  },
  "initial_inventory": {
    "27in 4K Gaming Monitor": 62440,
    "27in FHD Monitor": 75500
  },
  "decision_variables": {
    "x_1": "units of 27in 4K Gaming Monitor to fulfill",
    "x_2": "units of 27in FHD Monitor to fulfill"
  }
}