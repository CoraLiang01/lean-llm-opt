##### Decision Variables

$x_i \geq 0$: number of units of product $i$ that the company plans to fulfill, for each $i \in I$.

##### Objective Function

$\displaystyle \max \sum_{i\in I} r_i\,x_i$

##### Constraints

1. Demand satisfaction: $\displaystyle x_i \leq d_i,\quad \forall i\in I$  
2. Inventory limitation: $\displaystyle x_i \leq I_i,\quad \forall i\in I$  
3. Non‑negativity: $\displaystyle x_i \geq 0,\quad \forall i\in I$  

$I=\{P1,P2\}$, where $P1$ is “27in 4K Gaming Monitor” and $P2$ is “27in FHD Monitor”.

###### Retrieved Information

{
  "products": [
    "27in 4K Gaming Monitor",
    "27in FHD Monitor"
  ],
  "revenue": {
    "27in 4K Gaming Monitor": 389.99,
    "27in FHD Monitor": 149.99
  },
  "demand": {
    "27in 4K Gaming Monitor": 12474,
    "27in FHD Monitor": 15057
  },
  "initial_inventory": {
    "27in 4K Gaming Monitor": 62440,
    "27in FHD Monitor": 75500
  }
}