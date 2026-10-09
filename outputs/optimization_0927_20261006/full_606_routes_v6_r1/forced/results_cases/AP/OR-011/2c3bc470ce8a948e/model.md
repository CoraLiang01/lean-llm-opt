##### Objective Function:

$\quad \max \sum_{i \in \mathcal{P}} r_i x_i$

where:
- $\mathcal{P}$ is the set of products classified under ‘id999’ (here, a single product).
- $r_i$ is the revenue per unit for product $i$.
- $x_i$ is the number of units of product $i$ fulfilled.

##### Constraints:

1. Inventory and Demand Fulfillment Constraints:

$\quad 0 \leq x_i \leq \min\{\text{Initial Inventory}_i, \text{Demand}_i\} \quad \forall i \in \mathcal{P}$

2. Integer Constraints:

$\quad x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \mathcal{P}$

##### Retrieved Information

{
  "products": [
    {
      "id_number": "id999",
      "Revenue": 434.74,
      "Demand": 8171,
      "Initial Inventory": 56450
    }
  ]
}

##### Model with Parameters

Let $x_{id999}$ be the number of units of product id999 to fulfill.

Objective:
$$
\max\ 434.74\, x_{id999}
$$

Subject to:
$$
0 \leq x_{id999} \leq 8171
$$
$$
x_{id999} \in \mathbb{Z}_{\geq 0}
$$

All parameters (Revenue, Demand, Initial Inventory) are as retrieved above.