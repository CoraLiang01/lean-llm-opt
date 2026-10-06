Let $x_i$ denote the number of units of Baby product $i$ to fulfill.

Objective:
$$
\max \ 255.28 \cdot x_{\text{Baby Food\_255.28}}
$$

Subject to:
\begin{align*}
& x_{\text{Baby Food\_255.28}} \leq 3066513 \quad \text{(Demand constraint)} \\
& x_{\text{Baby Food\_255.28}} \leq 22749210 \quad \text{(Initial Inventory constraint)} \\
& x_{\text{Baby Food\_255.28}} \in \mathbb{Z}_{\geq 0}
\end{align*}

Where:
- $x_{\text{Baby Food\_255.28}}$ = number of units of "Baby Food_255.28" to fulfill (nonnegative integer)
- 255.28 = Revenue per unit
- 3066513 = Demand
- 22749210 = Initial Inventory