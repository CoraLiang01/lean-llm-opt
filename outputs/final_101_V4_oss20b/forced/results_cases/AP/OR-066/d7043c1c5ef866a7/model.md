##### Objective Function: 

$ \quad \min \sum_{i=1}^{2} \sum_{j=1}^{2} c_{ij} x_{ij} + \sum_{i=1}^{2} f_{i} y_{i} $

##### Constraints

###### 1. Demand Satisfaction Constraints: 

$ \sum_{i=1}^{2} x_{i1} = 144 \quad \forall j=1 $

$ \sum_{i=1}^{2} x_{i2} = 216 \quad \forall j=2 $

###### 2. Linking Constraints: 

$ x_{ij} \le 144 \, y_{i} \quad \forall i=1,2;\; j=1 $

$ x_{ij} \le 216 \, y_{i} \quad \forall i=1,2;\; j=2 $

###### 3. Variable Constraints: 

$ x_{ij} \ge 0 \quad \forall i=1,2;\; j=1,2 $

$ y_{i} \in \{0,1\} \quad \forall i=1,2 $

###### 4. Binary Constraints: 

$ y_{i} \in \{0,1\} \quad \forall i=1,2 $

###### 5. Non‑negativity Constraints: 

$ x_{ij} \ge 0 \quad \forall i=1,2;\; j=1,2 $

###### 6. Fixed Cost Activation Constraints: 

$ f_{i} y_{i} \ge 0 \quad \forall i=1,2 $

###### 7. Transportation Cost Constraints: 

$ c_{ij} x_{ij} \ge 0 \quad \forall i=1,2;\; j=1,2 $

###### 8. Demand Fulfillment Constraints: 

$ \sum_{i=1}^{2} x_{ij} = d_{j} \quad \forall j=1,2 $

###### 9. Supplier Activation Constraints: 

$ y_{i} \in \{0,1\} \quad \forall i=1,2 $

###### 10. Supply Capacity Constraints: 

$ x_{ij} \le M_{i} y_{i} \quad \forall i=1,2;\; j=1,2 $

##### Retrieved Information

{
  "fixed_costs": {
    "S1": 105.97,
    "S2": 85.31
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
  },
  "demand": {
    "C1": 144,
    "C2": 216
  },
  "cost_matrix": [
    [2358.39, 1492.08],
    [0.07, 52.32]
  ],
  "demand_vector": [144, 216],
  "fixed_cost_vector": [105.97, 85.31]
}