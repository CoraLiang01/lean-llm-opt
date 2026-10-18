##### Objective Function: 

$\quad \quad \max \sum_i p_i \cdot x_{i}$

##### Constraints

###### 1. Inventory Constraints: 

$\quad \quad \sum_i x_{i} \leq I_{\text{max}}, \quad \forall i$

###### 2. Demand Constraints: 

$\quad \quad \sum_i x_{i} \leq D_{\text{max}}, \quad \forall i$

###### 3. Non-negativity and Integrality Constraints: 

$\quad \quad  x_{i} \in \mathbb{Z}_{\geq 0}, \quad \forall i$

I indexes the ‘id999’ product(s); $p_i$ is Revenue, $I_{\text{max}}$ is Initial Inventory, $D_{\text{max}}$ is Demand. Variables are nonnegative integers representing whole product units.

###### Retrieved Information

{
  "items": [
    "id999"
  ],
  "profit": {
    "id999": 434.74
  },
  "demand": {
    "id999": 8171
  },
  "inventory": {
    "id999": 56450
  }
}