##### Decision Variables

$x_{\text{Aalopuri}} \in \mathbb{Z}_{\geq 0}$: Number of units of Aalopuri fulfilled.

##### Parameters

- Revenue per unit: $r_{\text{Aalopuri}} = 20$
- Demand: $d_{\text{Aalopuri}} = 1483$
- Initial Inventory: $I_{\text{Aalopuri}} = 10440$

##### Objective Function

$\max\ 20\, x_{\text{Aalopuri}}$

##### Constraints

1. Demand fulfillment: $x_{\text{Aalopuri}} \leq 1483$
2. Inventory limit: $x_{\text{Aalopuri}} \leq 10440$
3. Nonnegativity and integrality: $x_{\text{Aalopuri}} \in \mathbb{Z}_{\geq 0}$

##### Model Summary

Maximize total revenue from Aalopuri by choosing $x_{\text{Aalopuri}}$ units to fulfill, subject to available inventory and demand. All parameters are as retrieved from the data.