Let $i$ index the car models classified under ‘FDK57’, with the following data for each model $i$:

- $r_i$: Revenue for model $i$
- $d_i$: Demand for model $i$
- $s_i$: Initial Inventory for model $i$
- $x_i$: Number of units of model $i$ to fulfill (decision variable)

Given data (in source order):

| Product Name | Revenue  | Demand | Initial Inventory |
|--------------|----------|--------|------------------|
| FDK57        | 119.144  | 30     | 200              |
| FDK57        | 119.144  | 40     | 100              |
| FDK57        | 120.144  | 50     | 150              |

Decision variables:
$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i
$$

Objective:
$$
\max \sum_{i=1}^3 r_i x_i
$$

Subject to:

1. Demand fulfillment (cannot exceed demand):
$$
x_i \leq d_i, \quad \forall i
$$

2. Inventory constraint (cannot exceed initial inventory):
$$
x_i \leq s_i, \quad \forall i
$$

3. Non-negativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i
$$

Numerical formulation (in source order):

Let $x_1$ = units fulfilled for row 1, $x_2$ = units for row 2, $x_3$ = units for row 3.

Objective:
$$
\max \ 119.144\, x_1 + 119.144\, x_2 + 120.144\, x_3
$$

Subject to:
\begin{align*}
x_1 &\leq 30 \\
x_1 &\leq 200 \\
x_2 &\leq 40 \\
x_2 &\leq 100 \\
x_3 &\leq 50 \\
x_3 &\leq 150 \\
x_1, x_2, x_3 &\in \mathbb{Z}_{\geq 0}
\end{align*}