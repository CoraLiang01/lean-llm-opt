Let $i$ index the five FDK57 car model entries, in the order retrieved. Let $x_i$ denote the quantity of car model $i$ to fulfill.

Parameters (in source order):

\[
\begin{array}{llll}
i & \text{Revenue}_i & \text{Demand}_i & \text{InitialInventory}_i \\
1 & 119.144 & 30 & 200 \\
2 & 119.144 & 40 & 100 \\
3 & 120.144 & 50 & 150 \\
4 & 121.244 & 30 & 200 \\
5 & 120.844 & 50 & 150 \\
\end{array}
\]

Decision variables:

\[
x_i = \text{quantity of FDK57 car model $i$ to fulfill}, \quad x_i \in \mathbb{Z}_{\geq 0}
\]

Objective:

\[
\max \sum_{i=1}^5 \text{Revenue}_i \cdot x_i = 119.144\,x_1 + 119.144\,x_2 + 120.144\,x_3 + 121.244\,x_4 + 120.844\,x_5
\]

Subject to:

\[
\begin{align*}
& 0 \leq x_i \leq \min\{\text{Demand}_i,\, \text{InitialInventory}_i\}, \quad \forall i=1,\ldots,5 \\
& \text{That is:} \\
& 0 \leq x_1 \leq 30 \\
& 0 \leq x_2 \leq 40 \\
& 0 \leq x_3 \leq 50 \\
& 0 \leq x_4 \leq 30 \\
& 0 \leq x_5 \leq 50 \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i=1,\ldots,5
\end{align*}
\]

All variables and constraints are indexed in the original data order. No additional constraints are imposed.