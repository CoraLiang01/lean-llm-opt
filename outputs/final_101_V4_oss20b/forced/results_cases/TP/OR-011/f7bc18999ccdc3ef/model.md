##### Decision Variables

$x_i \in \mathbb{Z}_{\ge 0}$: number of units of product $i$ to fulfill.

##### Objective Function

$\max \sum_{i} r_i x_i$

##### Constraints

1. Demand: $\sum_{i} x_i \leq d_i,\quad \forall i$  
2. Inventory: $\sum_{i} x_i \leq s_i,\quad \forall i$  
3. Non‑negativity and integrality: $x_i \geq 0,\quad x_i \in \mathbb{Z},\quad \forall i$

$I=\{id999\}$, $J=\{id999\}$.

###### Retrieved Information

{
  "products": [
    {
      "id_number": "id999",
      "demand": 8171,
      "initial_inventory": 56450,
      "revenue": 434.74
    }
  ]
}