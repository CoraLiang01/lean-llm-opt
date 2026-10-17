##### Decision Variables

$x_i \geq 0$ (integer): number of units of ‘Baby Food_255.28’ to fulfill.

##### Parameters

- Revenue per unit: $r_1 = 255.28$
- Demand: $d_1 = 765{,}850$
- Initial Inventory: $I_1 = 5{,}627{,}060$

##### Objective Function

$\max\ 255.28\, x_1$

##### Constraints

1. Demand fulfillment: $x_1 \leq 765{,}850$
2. Inventory limit: $x_1 \leq 5{,}627{,}060$
3. Non-negativity and integrality: $x_1 \geq 0$, integer

##### Complete Model

$\begin{align*}
\max\ & 255.28\, x_1 \\
\text{s.t.}\quad
& x_1 \leq 765{,}850 \\
& x_1 \leq 5{,}627{,}060 \\
& x_1 \geq 0,\quad x_1 \in \mathbb{Z}
\end{align*}$

Where $x_1$ is the number of units of ‘Baby Food_255.28’ fulfilled.