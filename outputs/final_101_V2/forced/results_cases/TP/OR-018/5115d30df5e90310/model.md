##### Decision Variables

$x_i \geq 0$: number of units of Baby product $i$ to fulfill (continuous).

##### Parameters

- Product: Baby Food_255.28
- Revenue per unit: $255.28$
- Demand: $3,\!066,\!513$
- Initial Inventory: $22,\!749,\!210$

##### Objective Function

$\max\ 255.28\, x_{\text{Baby Food\_255.28}}$

##### Constraints

1. Demand fulfillment: $x_{\text{Baby Food\_255.28}} \leq 3,\!066,\!513$
2. Inventory limit: $x_{\text{Baby Food\_255.28}} \leq 22,\!749,\!210$
3. Non-negativity: $x_{\text{Baby Food\_255.28}} \geq 0$

##### Complete Model

\[
\begin{align*}
\max\quad & 255.28\, x_{\text{Baby Food\_255.28}} \\
\text{s.t.}\quad
& x_{\text{Baby Food\_255.28}} \leq 3,\!066,\!513 \\
& x_{\text{Baby Food\_255.28}} \leq 22,\!749,\!210 \\
& x_{\text{Baby Food\_255.28}} \geq 0
\end{align*}
\]