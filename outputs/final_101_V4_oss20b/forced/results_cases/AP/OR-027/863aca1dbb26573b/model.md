##### Objective Function: 

$\quad \quad \max \sum_{i=1}^{3} r_{i}\,x_{i}$

##### Constraints

###### 1. Demand Constraints: 

$\quad \quad x_{i} \leq d_{i}\quad \forall i \in \{1,2,3\}$

###### 2. Inventory Constraints: 

$\quad \quad x_{i} \leq I_{i}\quad \forall i \in \{1,2,3\}$

###### 3. Non‑negativity Constraints: 

$\quad \quad x_{i} \geq 0\quad \forall i \in \{1,2,3\}$

###### 4. Integer Constraints: 

$\quad \quad x_{i} \in \mathbb{Z}_{+}\quad \forall i \in \{1,2,3\}$

###### 5. Variable Definitions: 

$\quad \quad x_{i}$ denotes the number of units of Organ product $i$ that will be fulfilled.

###### 6. Parameter Definitions: 

$\quad \quad r_{i}$ – revenue per unit of Organ product $i$  
$\quad \quad d_{i}$ – demand quantity of Organ product $i$  
$\quad \quad I_{i}$ – initial inventory of Organ product $i$

###### 7. Index Mapping: 

$\quad \quad i=1 \rightarrow \text{Organic Fruits}$  
$\quad \quad i=2 \rightarrow \text{Organic Staples}$  
$\quad \quad i=3 \rightarrow \text{Organic Vegetables}$

###### 8. Retrieved Information

{
  "revenue": {
    "Organic Fruits": 60.8,
    "Organic Staples": 918.45,
    "Organic Vegetables": 77.52
  },
  "demand": {
    "Organic Fruits": 678906,
    "Organic Staples": 749927,
    "Organic Vegetables": 699808
  },
  "inventory": {
    "Organic Fruits": 5034020.0,
    "Organic Staples": 5589290.0,
    "Organic Vegetables": 5202710.0
  },
  "products": [
    "Organic Fruits",
    "Organic Staples",
    "Organic Vegetables"
  ]
}