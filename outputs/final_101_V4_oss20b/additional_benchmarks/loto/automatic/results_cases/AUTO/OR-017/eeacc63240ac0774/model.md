##### Objective Function: 

$ \quad \quad \max \sum_{i} \text{Revenue}_{i} \cdot x_{i}$

##### Constraints

###### 1. Demand Constraints: 

$ \quad \quad x_{i} \leq \text{Demand}_{i}, \quad \forall i$

###### 2. Initial Inventory Constraints: 

$ \quad \quad x_{i} \leq \text{InitialInventory}_{i}, \quad \forall i$

###### 3. Non-negativity and Integrality Constraints: 

$ \quad \quad x_{i} \in \mathbb{Z}_{\geq 0}, \quad \forall i$

###### Retrieved Information

{
  "items": ["ZZ2AO", "ZZDW7", "ZZM1A", "ZZNC5", "ZZX6K"],
  "revenue": {
    "ZZ2AO": 24.38,
    "ZZDW7": 30.12,
    "ZZM1A": 19.52,
    "ZZNC5": 10.79,
    "ZZX6K": 111.81
  },
  "demand": {
    "ZZ2AO": 2,
    "ZZDW7": 4,
    "ZZM1A": 82,
    "ZZNC5": 2,
    "ZZX6K": 2
  },
  "initial_inventory": {
    "ZZ2AO": 10.0,
    "ZZDW7": 20.0,
    "ZZM1A": 530.0,
    "ZZNC5": 10.0,
    "ZZX6K": 10.0
  }
}