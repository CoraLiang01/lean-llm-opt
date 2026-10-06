Let $x_i$ denote the number of units of product $i$ (here, "Baby Food_255.28") to be fulfilled.

Objective:
$$
\max\ 255.28\, x_{\text{Baby Food\_255.28}}
$$

Subject to:
\[
\begin{align*}
& x_{\text{Baby Food\_255.28}} \leq 765850 \quad \text{(Demand constraint)} \\
& x_{\text{Baby Food\_255.28}} \leq 5627060 \quad \text{(Initial Inventory constraint)} \\
& x_{\text{Baby Food\_255.28}} \in \mathbb{Z}_{\geq 0}
\end{align*}
\]

Where:
- $x_{\text{Baby Food\_255.28}}$ = number of units of "Baby Food_255.28" fulfilled
- 255.28 = Revenue per unit
- 765850 = Demand
- 5627060 = Initial Inventory