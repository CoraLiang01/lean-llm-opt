##### Objective Function: 

$ \quad \quad \max \sum_{i=1}^{5} r_{i} x_{i} $

##### Constraints

###### 1. Demand Constraints: 

$ x_{i} \leq d_{i} \quad \forall i \in \{1,2,3,4,5\} $

###### 2. Inventory Constraints: 

$ x_{i} \leq I_{i} \quad \forall i \in \{1,2,3,4,5\} $

###### 3. Non‑negativity Constraints: 

$ x_{i} \geq 0 \quad \forall i \in \{1,2,3,4,5\} $

###### 4. Integer Constraints (if required): 

$ x_{i} \in \mathbb{Z} \quad \forall i \in \{1,2,3,4,5\} $

##### Retrieved Information

{
  "products": [
    {
      "name": "Books_15.15",
      "revenue": 15.15,
      "demand": 1980,
      "inventory": 9920.0
    },
    {
      "name": "Books_30.3",
      "revenue": 30.3,
      "demand": 3024,
      "inventory": 20160.0
    },
    {
      "name": "Books_45.45",
      "revenue": 45.45,
      "demand": 4536,
      "inventory": 30000.0
    },
    {
      "name": "Books_60.6",
      "revenue": 60.6,
      "demand": 5601,
      "inventory": 38360.0
    },
    {
      "name": "Books_75.75",
      "revenue": 75.75,
      "demand": 7567,
      "inventory": 51450.0
    }
  ],
  "vectors": {
    "revenue": [15.15, 30.3, 45.45, 60.6, 75.75],
    "demand": [1980, 3024, 4536, 5601, 7567],
    "inventory": [9920.0, 20160.0, 30000.0, 38360.0, 51450.0]
  }
}