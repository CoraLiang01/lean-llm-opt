##### Objective Function:

$\quad \min \left( \sum_{i=1}^2 f_i y_i + \sum_{i=1}^2 \sum_{j=1}^2 c_{ij} x_{ij} \right)$

where:
- $f_1 = 105.97$ (fixed cost for supplier S1)
- $f_2 = 85.31$ (fixed cost for supplier S2)
- $c_{11} = 2358.39$ (transportation cost from S1 to C1)
- $c_{12} = 1492.08$ (transportation cost from S1 to C2)
- $c_{21} = 0.07$ (transportation cost from S2 to C1)
- $c_{22} = 52.32$ (transportation cost from S2 to C2)

##### Constraints

###### 1. Demand Satisfaction:

$\quad x_{1j} + x_{2j} = d_j \quad \forall j \in \{1,2\}$

where:
- $d_1 = 144$ (demand at C1)
- $d_2 = 216$ (demand at C2)

###### 2. Supplier Activation:

$\quad x_{ij} \leq d_j y_i \quad \forall i \in \{1,2\},\ \forall j \in \{1,2\}$

###### 3. Variable Domains:

$\quad y_i \in \{0,1\} \quad \forall i \in \{1,2\}$

$\quad x_{ij} \geq 0 \quad \forall i \in \{1,2\},\ \forall j \in \{1,2\}$

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