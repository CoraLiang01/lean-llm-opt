Let $x_i$ be the quantity of FDK57 car model $i$ to fulfill, for each row $i$ below.

Objective:
$$
\max\ 119.144\, x_1 + 120.144\, x_2 + 121.244\, x_3
$$

Subject to:
\[
\begin{align*}
& 0 \leq x_1 \leq \min\{200,\ 30\} \\
& 0 \leq x_2 \leq \min\{150,\ 50\} \\
& 0 \leq x_3 \leq \min\{150,\ 30\} \\
& x_1, x_2, x_3 \in \mathbb{Z}_{\geq 0}
\end{align*}
\]

Where:
- $x_1$ = quantity fulfilled for FDK57 (row 1: Revenue 119.144, Initial Inventory 200, Demand 30)
- $x_2$ = quantity fulfilled for FDK57 (row 2: Revenue 120.144, Initial Inventory 150, Demand 50)
- $x_3$ = quantity fulfilled for FDK57 (row 3: Revenue 121.244, Initial Inventory 150, Demand 30)

Each $x_i$ is a nonnegative integer, bounded above by both the initial inventory and demand for that row.