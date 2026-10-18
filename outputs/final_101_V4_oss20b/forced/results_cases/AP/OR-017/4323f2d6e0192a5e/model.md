##### Objective Function: 

$ \quad \quad \max \sum_{i=1}^{5} rev_{i}\,x_{i} $

##### Constraints

###### 1. Demand Constraints: 

$ x_{i} \leq dem_{i} \quad \forall i \in \{1,2,3,4,5\} $

###### 2. Inventory Constraints: 

$ x_{i} \leq inv_{i} \quad \forall i \in \{1,2,3,4,5\} $

###### 3. Non‑negativity Constraints: 

$ x_{i} \geq 0 \quad \forall i \in \{1,2,3,4,5\} $

###### 4. Integer Constraints: 

$ x_{i} \in \mathbb{Z}_{+} \quad \forall i \in \{1,2,3,4,5\} $

###### 5. Variable Definitions: 

$ x_{i} $ represents the number of units of ZZ product $i$ to be fulfilled.

###### 6. Parameter Definitions: 

$ rev_{i} $ – revenue per unit of ZZ product $i$  
$ dem_{i} $ – demand quantity of ZZ product $i$  
$ inv_{i} $ – initial inventory of ZZ product $i$

###### 7. Parameter Values: 

$ rev_{1}=24.38,\; dem_{1}=2,\; inv_{1}=10.0 $  
$ rev_{2}=30.12,\; dem_{2}=4,\; inv_{2}=20.0 $  
$ rev_{3}=19.52,\; dem_{3}=82,\; inv_{3}=530.0 $  
$ rev_{4}=10.79,\; dem_{4}=2,\; inv_{4}=10.0 $  
$ rev_{5}=111.81,\; dem_{5}=2,\; inv_{5}=10.0 $

###### 8. Retrieved Information

{
  "SKU": [
    "ZZ2AO",
    "ZZDW7",
    "ZZM1A",
    "ZZNC5",
    "ZZX6K"
  ],
  "Revenue": [
    24.38,
    30.12,
    19.52,
    10.79,
    111.81
  ],
  "Demand": [
    2,
    4,
    82,
    2,
    2
  ],
  "Initial Inventory": [
    10.0,
    20.0,
    530.0,
    10.0,
    10.0
  ]
}