Let $i$ index the six FDK57 car model entries in the order retrieved. Let $x_i$ be the number of units of FDK57 car model $i$ to fulfill (sell).

Parameters (from the data, in source order):

\[
\begin{array}{cccc}
i & \text{Revenue}_i & \text{Demand}_i & \text{InitialInventory}_i \\
1 & 119.144 & 30 & 200 \\
2 & 119.144 & 40 & 100 \\
3 & 120.144 & 50 & 150 \\
4 & 121.244 & 30 & 200 \\
5 & 120.544 & 10 & 150 \\
6 & 121.244 & 30 & 150 \\
\end{array}
\]

Decision variables:

\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i=1,\ldots,6
\]

Objective:

\[
\max \sum_{i=1}^6 \text{Revenue}_i \cdot x_i
\]

Subject to:

\[
\begin{align*}
& x_i \leq \text{Demand}_i, \quad \forall i=1,\ldots,6 \\
& x_i \leq \text{InitialInventory}_i, \quad \forall i=1,\ldots,6 \\
& x_i \geq 0 \text{ and integer}, \quad \forall i=1,\ldots,6
\end{align*}
\]

Explicitly, the model is:

\[
\begin{align*}
\max\ & 119.144\,x_1 + 119.144\,x_2 + 120.144\,x_3 + 121.244\,x_4 + 120.544\,x_5 + 121.244\,x_6 \\
\text{s.t.}\quad
& x_1 \leq 30 \\
& x_2 \leq 40 \\
& x_3 \leq 50 \\
& x_4 \leq 30 \\
& x_5 \leq 10 \\
& x_6 \leq 30 \\
& x_1 \leq 200 \\
& x_2 \leq 100 \\
& x_3 \leq 150 \\
& x_4 \leq 200 \\
& x_5 \leq 150 \\
& x_6 \leq 150 \\
& x_i \in \mathbb{Z}_{\geq 0},\quad i=1,\ldots,6
\end{align*}
\]