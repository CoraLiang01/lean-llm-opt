Let $x_i$ denote the quantity of each FDK57 car model $i$ to fulfill, for each row $i$ in the data below.

Maximize total revenue:
$$
\max\ 119.144\, x_1 + 119.144\, x_2 + 120.144\, x_3
$$

Subject to:

Demand constraints:
\[
\begin{align*}
x_1 &\leq 30 \\
x_2 &\leq 40 \\
x_3 &\leq 50 \\
\end{align*}
\]

Inventory constraints:
\[
\begin{align*}
x_1 &\leq 200 \\
x_2 &\leq 100 \\
x_3 &\leq 150 \\
\end{align*}
\]

Nonnegativity and integrality:
\[
x_1,\, x_2,\, x_3 \in \mathbb{Z}_{\geq 0}
\]

Where:

| Row | Product Name | Revenue  | Demand | Initial Inventory |
|-----|--------------|----------|--------|------------------|
| 1   | FDK57        | 119.144  | 30     | 200              |
| 2   | FDK57        | 119.144  | 40     | 100              |
| 3   | FDK57        | 120.144  | 50     | 150              |

Each $x_i$ represents the number of units of the corresponding FDK57 car model to fulfill, subject to both demand and inventory limits.