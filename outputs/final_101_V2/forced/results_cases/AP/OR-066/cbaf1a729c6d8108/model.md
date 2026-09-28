##### Objective Function:

$\quad \min \left( \sum_{i \in \{S1, S2\}} \text{fixed\_costs}_i \cdot y_i + \sum_{i \in \{S1, S2\}} \sum_{j \in \{C1, C2\}} \text{transport\_cost}_{ij} \cdot x_{ij} \right)$

##### Constraints

###### 1. Demand Satisfaction:

$\sum_{i \in \{S1, S2\}} x_{ij} = \text{demand}_j \quad \forall j \in \{C1, C2\}$

###### 2. Supplier Activation:

$\sum_{j \in \{C1, C2\}} x_{ij} \leq \left( \sum_{j \in \{C1, C2\}} \text{demand}_j \right) \cdot y_i \quad \forall i \in \{S1, S2\}$

###### 3. Variable Domains:

$y_i \in \{0,1\} \quad \forall i \in \{S1, S2\}$

$x_{ij} \geq 0 \quad \forall i \in \{S1, S2\},\ j \in \{C1, C2\}$

##### Retrieved Information

{
  "fixed_costs": {
    "S1": 105.97,
    "S2": 85.31
  },
  "transport_cost": {
    "S1": {
      "C1": 2358.39,
      "C2": 1492.08
    },
    "S2": {
      "C1": 0.07,
      "C2": 52.32
    }
  },
  "demand": {
    "C1": 144,
    "C2": 216
  },
  "suppliers": ["S1", "S2"],
  "customers": ["C1", "C2"]
}