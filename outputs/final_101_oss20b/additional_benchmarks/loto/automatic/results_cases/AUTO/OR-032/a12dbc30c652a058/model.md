##### Objective Function: 

$ \max \sum_{i} p_i \cdot x_i $

##### Constraints

###### 1. Demand Constraints: 

$ x_i \le d_i, \quad \forall i $

###### 2. Inventory Constraints: 

$ x_i \le s_i, \quad \forall i $

###### 3. Non‑negativity and Integrality Constraints: 

$ x_i \in \mathbb{Z}_{\ge 0}, \quad \forall i $

Indices: $i$ indexes the five Books products.  
$p_i$ is the revenue per unit, $d_i$ is the demand, and $s_i$ is the initial inventory.

###### Retrieved Information

{
  "items": [
    "Books_15.15",
    "Books_30.3",
    "Books_45.45",
    "Books_60.6",
    "Books_75.75"
  ],
  "profit": {
    "Books_15.15": 15.15,
    "Books_30.3": 30.3,
    "Books_45.45": 45.45,
    "Books_60.6": 60.6,
    "Books_75.75": 75.75
  },
  "demand": {
    "Books_15.15": 1980,
    "Books_30.3": 3024,
    "Books_45.45": 4536,
    "Books_60.6": 5601,
    "Books_75.75": 7567
  },
  "inventory": {
    "Books_15.15": 9920.0,
    "Books_30.3": 20160.0,
    "Books_45.45": 30000.0,
    "Books_60.6": 38360.0,
    "Books_75.75": 51450.0
  }
}