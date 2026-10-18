Let $x_k$ denote the quantity of ‘FDK57’ car model in row $k$ to fulfill, for $k=1,\ldots,5$ (corresponding to the five retrieved rows, in source order).

Maximize total revenue:
$$
\max\ 119.144\,x_1 + 119.144\,x_2 + 121.244\,x_3 + 120.144\,x_4 + 120.844\,x_5
$$

Subject to, for each $k=1,\ldots,5$:
\[
\begin{align*}
& x_1 \leq 30 \\
& x_1 \leq 200 \\
& x_2 \leq 40 \\
& x_2 \leq 100 \\
& x_3 \leq 30 \\
& x_3 \leq 200 \\
& x_4 \leq 50 \\
& x_4 \leq 150 \\
& x_5 \leq 50 \\
& x_5 \leq 150 \\
& x_k \geq 0,\quad x_k \in \mathbb{Z},\quad \forall k=1,\ldots,5
\end{align*}
\]

Where:
- $x_1$ = quantity fulfilled for row 1: Revenue 119.144, Demand 30, Initial Inventory 200
- $x_2$ = quantity fulfilled for row 2: Revenue 119.144, Demand 40, Initial Inventory 100
- $x_3$ = quantity fulfilled for row 3: Revenue 121.244, Demand 30, Initial Inventory 200
- $x_4$ = quantity fulfilled for row 4: Revenue 120.144, Demand 50, Initial Inventory 150
- $x_5$ = quantity fulfilled for row 5: Revenue 120.844, Demand 50, Initial Inventory 150

That is, for each row $k$, $x_k$ is bounded above by both its Demand and Initial Inventory, and must be a nonnegative integer.