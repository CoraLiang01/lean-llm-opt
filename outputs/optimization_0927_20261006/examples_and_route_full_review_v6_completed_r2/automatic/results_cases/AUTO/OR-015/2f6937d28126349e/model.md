Let $x_i$ denote the number of units of Aalop product $i$ to fulfill.

Objective:
$$
\max \ 20 \cdot x_{\text{Aalopuri}}
$$

Subject to:
\[
\begin{align*}
& x_{\text{Aalopuri}} \leq 10440.0 \quad \text{(Initial Inventory constraint)} \\
& x_{\text{Aalopuri}} \leq 1483 \quad \text{(Demand constraint)} \\
& x_{\text{Aalopuri}} \geq 0 \\
& x_{\text{Aalopuri}} \in \mathbb{Z}
\end{align*}
\]

Where:
- $x_{\text{Aalopuri}}$ = number of Aalopuri units fulfilled (integer, nonnegative)
- Revenue per unit: 20
- Initial Inventory: 10440.0
- Demand: 1483