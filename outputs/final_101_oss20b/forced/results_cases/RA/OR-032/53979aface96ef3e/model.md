##### Objective Function: 

$\quad \quad \max \sum_{i=1}^{5} p_i \cdot x_i$

##### Constraints

###### 1. Demand Constraints: 

$\quad \quad x_i \leq d_i, \quad \forall i$

###### 2. Inventory Constraints: 

$\quad \quad x_i \leq s_i, \quad \forall i$

###### 3. Non-negativity and Integrality Constraints: 

$\quad \quad x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i$

##### Retrieved Information

{
  "items": [
    "Books_15.15",
    "Books_30.3",
    "Books_45.45",
    "Books_60.6",
    "Books_75.75"
  ],
  "revenue": {
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