##### Objective Function: 

$\quad \quad \max \sum_{i=1}^{3} r_{i} x_{i}$

##### Constraints

###### 1. Demand Constraints: 

$\quad \quad x_{i} \leq d_{i} \quad \forall i \in \{1,2,3\}$

###### 2. Inventory Constraints: 

$\quad \quad x_{i} \leq I_{i} \quad \forall i \in \{1,2,3\}$

###### 3. Non‑negativity Constraints: 

$\quad \quad x_{i} \geq 0 \quad \forall i \in \{1,2,3\}$

###### 4. Integer Constraints: 

$\quad \quad x_{i} \in \mathbb{Z} \quad \forall i \in \{1,2,3\}$

###### 5. Variable Definitions: 

$\quad \quad x_{i}$ represents the number of units of product $i$ to be fulfilled.

###### 6. Parameter Definitions: 

$\quad \quad r_{i}$ is the revenue per unit of product $i$;  
$\quad \quad d_{i}$ is the demand for product $i$;  
$\quad \quad I_{i}$ is the initial inventory of product $i$.

###### 7. Index Mapping: 

$\quad \quad i=1 \rightarrow \text{4U\_Service\_22}$  
$\quad \quad i=2 \rightarrow \text{4U\_Service\_36}$  
$\quad \quad i=3 \rightarrow \text{4U\_Service\_7}$

###### 8. Parameter Values: 

$\quad \quad r_{1}=56.0,\; d_{1}=5,\; I_{1}=30$  
$\quad \quad r_{2}=21.6,\; d_{2}=3,\; I_{2}=20$  
$\quad \quad r_{3}=62.5,\; d_{3}=3,\; I_{3}=20$

###### 9. Retrieved Information

{
  "revenue": {
    "4U_Service_22": 56.0,
    "4U_Service_36": 21.6,
    "4U_Service_7": 62.5
  },
  "demand": {
    "4U_Service_22": 5,
    "4U_Service_36": 3,
    "4U_Service_7": 3
  },
  "inventory": {
    "4U_Service_22": 30,
    "4U_Service_36": 20,
    "4U_Service_7": 20
  },
  "products": [
    "4U_Service_22",
    "4U_Service_36",
    "4U_Service_7"
  ]
}