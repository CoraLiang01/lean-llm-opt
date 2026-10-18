##### Objective Function: 

$ \quad \quad \max \sum_{i \in I} r_i \cdot x_i $

##### Constraints

###### 1. Demand Constraints: 

$ \quad \quad x_i \leq D_i, \quad \forall i \in I $

###### 2. Inventory Constraints: 

$ \quad \quad x_i \leq I_i, \quad \forall i \in I $

###### 3. Non-negativity and Integrality Constraints: 

$ \quad \quad x_i \in \mathbb{Z}_{\ge 0}, \quad \forall i \in I $

I indexes the five ZZ products; I = {ZZ2AO, ZZDW7, ZZM1A, ZZNC5, ZZX6K}. r_i is Revenue, D_i is Demand, I_i is Initial Inventory. Variables are nonnegative integers representing whole product units.

###### Retrieved Information

{
  "items": [
    "ZZ2AO",
    "ZZDW7",
    "ZZM1A",
    "ZZNC5",
    "ZZX6K"
  ],
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