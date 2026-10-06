##### Sets and Indices
Let $i$ index the following product:
- Product Name: Baby Food_255.28

##### Parameters (from data)
- Revenue per unit: $r_i = 255.28$
- Demand: $d_i = 765{,}850$
- Initial Inventory: $s_i = 5{,}627{,}060$

##### Decision Variables
- $x_i$: Number of units of product $i$ to fulfill (nonnegative integer)

##### Mathematical Model

Objective:
$$
\max \ 255.28 \cdot x_i
$$

Subject to:
$$
x_i \leq 5{,}627{,}060 \qquad \text{(Inventory constraint)}
$$
$$
x_i \leq 765{,}850 \qquad \text{(Demand constraint)}
$$
$$
x_i \in \mathbb{Z}_{\geq 0}
$$

Where:
- $x_i$ is the number of units of Baby Food_255.28 fulfilled.