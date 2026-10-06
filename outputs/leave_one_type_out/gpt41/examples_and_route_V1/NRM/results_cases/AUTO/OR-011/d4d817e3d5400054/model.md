Let $x_{i}$ denote the number of units of product $i$ (with classification id999) to fulfill.

Objective:
$$
\max\ 434.74\, x_{id999}
$$

Subject to:
$$
x_{id999} \leq 56450 \quad \text{(Initial Inventory constraint)}
$$
$$
x_{id999} \leq 8171 \quad \text{(Demand constraint)}
$$
$$
x_{id999} \in \mathbb{Z}_{\geq 0}
$$

Where:
- $x_{id999}$: Number of units of product id999 to fulfill (decision variable, non-negative integer)
- 434.74: Revenue per unit of product id999
- 56450: Initial inventory of product id999
- 8171: Demand for product id999 during the sales horizon

The model maximizes total revenue from fulfilling product id999, subject to inventory and demand limits.