Let $x_i$ denote the quantity of FDK57 car model $i$ to fulfill, for $i=1,\ldots,5$, corresponding to the five rows below in source order.

Objective:
$$
\max\ 119.144\, x_1 + 119.144\, x_2 + 120.144\, x_3 + 121.244\, x_4 + 120.544\, x_5
$$

Subject to:
\[
\begin{align*}
& 0 \leq x_1 \leq \min\{200,\,30\} = 30 \\
& 0 \leq x_2 \leq \min\{100,\,40\} = 40 \\
& 0 \leq x_3 \leq \min\{150,\,50\} = 50 \\
& 0 \leq x_4 \leq \min\{200,\,30\} = 30 \\
& 0 \leq x_5 \leq \min\{150,\,10\} = 10 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,5
\end{align*}
\]

Where:
- $x_i$ = quantity of FDK57 car model $i$ fulfilled (integer, nonnegative)
- The coefficients and bounds are taken directly from the retrieved data, in source order:

| $i$ | Revenue   | Demand | Initial Inventory | $x_i$ upper bound |
|-----|-----------|--------|------------------|-------------------|
| 1   | 119.144   | 30     | 200              | 30                |
| 2   | 119.144   | 40     | 100              | 40                |
| 3   | 120.144   | 50     | 150              | 50                |
| 4   | 121.244   | 30     | 200              | 30                |
| 5   | 120.544   | 10     | 150              | 10                |

All variables are nonnegative integers, and each $x_i$ cannot exceed both the initial inventory and demand for its row. The objective is to maximize total revenue from fulfilled quantities.