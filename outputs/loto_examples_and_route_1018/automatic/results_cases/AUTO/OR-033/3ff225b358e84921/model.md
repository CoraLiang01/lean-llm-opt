Let $x_i$ denote the number of units of each ‘Baby’ product $i$ to be fulfilled.

Given data (from the retrieved row):

- Product Name: Baby Food_255.28
- Revenue: 255.28
- Demand: 765850
- Initial Inventory: 5627060

Mathematical Model:

Objective:
$$
\max\ 255.28\, x_{\text{Baby Food\_255.28}}
$$

Subject to:
\[
\begin{align*}
& x_{\text{Baby Food\_255.28}} \leq 765850 \quad \text{(Demand constraint)} \\
& x_{\text{Baby Food\_255.28}} \leq 5627060 \quad \text{(Inventory constraint)} \\
& x_{\text{Baby Food\_255.28}} \geq 0 \\
& x_{\text{Baby Food\_255.28}} \in \mathbb{Z}
\end{align*}
\]

Where $x_{\text{Baby Food\_255.28}}$ is the number of units of "Baby Food_255.28" to be fulfilled.