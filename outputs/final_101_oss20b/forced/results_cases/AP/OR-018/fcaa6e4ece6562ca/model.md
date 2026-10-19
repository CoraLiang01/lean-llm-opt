##### Objective Function: 

$ \quad \max \sum_{i=1}^{1} r_{i}\,x_{i} $

##### Constraints

###### 1. Demand Constraints: 

$\forall i \in \{1\}:\; x_{i}\;\leq\;d_{i}$

###### 2. Inventory Constraints: 

$\forall i \in \{1\}:\; x_{i}\;\leq\;I_{i}$

###### 3. Non‑negativity and Integrality Constraints: 

$x_{i}\;\in\;\mathbb{Z}_{+}\;\;\forall i \in \{1\}$

##### Retrieved Information

{
  "products": [
    {
      "Product Name": "Baby Food_255.28",
      "Revenue": 255.28,
      "Demand": 3066513,
      "Initial Inventory": 22749210
    }
  ],
  "variables": {
    "x": [
      {
        "index": 1,
        "description": "Units of Baby Food_255.28 to be sold"
      }
    ]
  },
  "parameters": {
    "r": [
      {
        "index": 1,
        "value": 255.28
      }
    ],
    "d": [
      {
        "index": 1,
        "value": 3066513
      }
    ],
    "I": [
      {
        "index": 1,
        "value": 22749210
      }
    ]
  }
}