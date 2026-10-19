##### Objective Function: 

$\quad \quad \max \sum_{i=1}^{8} p_{i} \cdot x_{i}$

##### Constraints

###### 1. Demand Constraints: 

$\quad \quad x_{i} \leq d_{i}, \quad \forall i=1,\dots ,8$

###### 2. Inventory Constraints: 

$\quad \quad x_{i} \leq I_{i}, \quad \forall i=1,\dots ,8$

###### 3. Non‑negativity and Integrality Constraints: 

$\quad \quad x_{i} \in \mathbb{Z}_{\geq 0}, \quad \forall i=1,\dots ,8$

Where $p_{i}$ is the revenue per unit of model $i$, $d_{i}$ is the demand for model $i$, and $I_{i}$ is the initial inventory for model $i$.

###### Retrieved Information

{
  "items": [
    "FDK57_1",
    "FDK57_2",
    "FDK57_3",
    "FDK57_4",
    "FDK57_5",
    "FDK57_6",
    "FDK57_7",
    "FDK57_8"
  ],
  "revenue": {
    "FDK57_1": 119.144,
    "FDK57_2": 119.144,
    "FDK57_3": 121.244,
    "FDK57_4": 120.144,
    "FDK57_5": 120.544,
    "FDK57_6": 121.244,
    "FDK57_7": 119.744,
    "FDK57_8": 120.844
  },
  "demand": {
    "FDK57_1": 30,
    "FDK57_2": 40,
    "FDK57_3": 30,
    "FDK57_4": 50,
    "FDK57_5": 10,
    "FDK57_6": 30,
    "FDK57_7": 50,
    "FDK57_8": 50
  },
  "inventory": {
    "FDK57_1": 200,
    "FDK57_2": 100,
    "FDK57_3": 200,
    "FDK57_4": 150,
    "FDK57_5": 150,
    "FDK57_6": 150,
    "FDK57_7": 250,
    "FDK57_8": 150
  }
}