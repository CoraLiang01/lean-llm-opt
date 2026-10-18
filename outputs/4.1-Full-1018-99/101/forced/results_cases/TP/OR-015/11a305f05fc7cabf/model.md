##### Decision Variables

Let $x_{\text{Aalopuri}} \geq 0$ (integer): number of Aalopuri units to fulfill.

##### Parameters

- Revenue per unit: $r_{\text{Aalopuri}} = 20$
- Demand: $d_{\text{Aalopuri}} = 1483$
- Initial Inventory: $I_{\text{Aalopuri}} = 10440.0$

##### Objective

$\max\ 20\, x_{\text{Aalopuri}}$

##### Constraints

$0 \leq x_{\text{Aalopuri}} \leq \min\{1483,\ 10440.0\}$

$x_{\text{Aalopuri}}$ integer

##### Complete Model

\[
\begin{align*}
\max\quad & 20\, x_{\text{Aalopuri}} \\
\text{s.t.}\quad & 0 \leq x_{\text{Aalopuri}} \leq 1483 \\
                 & x_{\text{Aalopuri}} \leq 10440.0 \\
                 & x_{\text{Aalopuri}} \in \mathbb{Z}_{\geq 0}
\end{align*}
\]

where $x_{\text{Aalopuri}}$ is the number of Aalopuri units fulfilled, subject to demand and initial inventory limits.