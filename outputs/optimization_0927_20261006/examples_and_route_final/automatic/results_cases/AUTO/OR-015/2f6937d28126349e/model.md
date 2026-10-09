Let $x_{\text{Aalopuri}}$ be the number of units of Aalopuri to fulfill.

Objective:
$$
\max\ 20\, x_{\text{Aalopuri}}
$$

Subject to:
\[
\begin{align*}
& 0 \leq x_{\text{Aalopuri}} \leq 1483 \\
& x_{\text{Aalopuri}} \leq 10440 \\
& x_{\text{Aalopuri}} \in \mathbb{Z}_{\geq 0}
\end{align*}
\]

Where:
- $x_{\text{Aalopuri}}$ = number of Aalopuri units fulfilled (decision variable, nonnegative integer)
- 20 = Revenue per unit (from ‘Revenue’ column)
- 1483 = Demand for Aalopuri (from ‘Demand’ column)
- 10440 = Initial Inventory for Aalopuri (from ‘Initial Inventory’ column)

All constraints and coefficients are taken directly from the retrieved data.