Let $i$ index the five FDK57 car model entries in the order retrieved.

Let $x_i$ = number of units of FDK57 car model $i$ to fulfill (nonnegative integer).

Parameters (in source order):

\[
\begin{array}{cccc}
i & \text{Revenue}_i & \text{Demand}_i & \text{InitialInventory}_i \\
1 & 119.144 & 30 & 200 \\
2 & 119.144 & 40 & 100 \\
3 & 121.244 & 30 & 200 \\
4 & 120.144 & 50 & 150 \\
5 & 120.844 & 50 & 150 \\
\end{array}
\]

Objective:
\[
\max \; 119.144\, x_1 + 119.144\, x_2 + 121.244\, x_3 + 120.144\, x_4 + 120.844\, x_5
\]

Subject to:
\[
\begin{align*}
& 0 \leq x_1 \leq \min\{30,\,200\} = 30 \\
& 0 \leq x_2 \leq \min\{40,\,100\} = 40 \\
& 0 \leq x_3 \leq \min\{30,\,200\} = 30 \\
& 0 \leq x_4 \leq \min\{50,\,150\} = 50 \\
& 0 \leq x_5 \leq \min\{50,\,150\} = 50 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,5
\end{align*}
\]

Or, explicitly:
\[
\begin{align*}
& 0 \leq x_1 \leq 30 \\
& 0 \leq x_2 \leq 40 \\
& 0 \leq x_3 \leq 30 \\
& 0 \leq x_4 \leq 50 \\
& 0 \leq x_5 \leq 50 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,5
\end{align*}
\]

Where:
- $x_i$ = quantity of FDK57 car model $i$ to fulfill (integer, $0 \leq x_i \leq$ both demand and initial inventory for $i$)
- Revenue, Demand, and Initial Inventory are as listed above for each $i$ in source order.