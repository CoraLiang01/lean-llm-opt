##### Objective Function:

$\quad \min \left( \sum_{i \in \{S1, S2\}} f_i y_i + \sum_{i \in \{S1, S2\}} \sum_{j \in \{C1, C2\}} c_{ij} x_{ij} \right)$

where:
- $f_i$ is the fixed cost for supplier $i$
- $c_{ij}$ is the per-unit transportation cost from supplier $i$ to customer $j$
- $y_i$ is a binary variable indicating if supplier $i$ is activated
- $x_{ij}$ is the amount supplied from supplier $i$ to customer $j$

##### Constraints

###### 1. Demand Satisfaction:

$\sum_{i \in \{S1, S2\}} x_{ij} = d_j \quad \forall j \in \{C1, C2\}$

###### 2. Supplier Activation:

$x_{ij} \leq d_j y_i \quad \forall i \in \{S1, S2\}, \forall j \in \{C1, C2\}$

###### 3. Variable Domains:

$y_i \in \{0,1\} \quad \forall i \in \{S1, S2\}$

$x_{ij} \geq 0 \quad \forall i \in \{S1, S2\}, \forall j \in \{C1, C2\}$

##### Retrieved Information

{
  "suppliers": [
    "S1",
    "S2"
  ],
  "customers": [
    "C1",
    "C2"
  ],
  "fixed_costs": {
    "S1": 105.97,
    "S2": 85.31
  },
  "demands": {
    "C1": 144,
    "C2": 216
  },
  "transportation_costs": {
    "S1": {
      "C1": 2358.39,
      "C2": 1492.08
    },
    "S2": {
      "C1": 0.07,
      "C2": 52.32
    }
  }
}