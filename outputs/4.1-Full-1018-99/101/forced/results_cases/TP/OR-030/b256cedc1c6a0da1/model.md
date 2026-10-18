Let $i$ index the car models classified under ‘FDK57’ as they appear in the data (source order). Let $x_i$ be the quantity of car model $i$ to fulfill (decision variable, continuous and nonnegative).

Parameters (in source order):

\[
\begin{array}{llll}
\text{Model} & \text{Revenue}_i & \text{Demand}_i & \text{InitialInventory}_i \\
1 & 119.144 & 30 & 200 \\
2 & 119.144 & 40 & 100 \\
3 & 120.144 & 50 & 150 \\
4 & 121.244 & 30 & 200 \\
\end{array}
\]

Decision variables:

\[
x_i \geq 0 \quad \text{(quantity of car model $i$ fulfilled), for } i=1,2,3,4
\]

Objective:

\[
\max \; 119.144\,x_1 + 119.144\,x_2 + 120.144\,x_3 + 121.244\,x_4
\]

Subject to:

\[
\begin{align*}
& 0 \leq x_1 \leq \min\{30,\,200\} = 30 \\
& 0 \leq x_2 \leq \min\{40,\,100\} = 40 \\
& 0 \leq x_3 \leq \min\{50,\,150\} = 50 \\
& 0 \leq x_4 \leq \min\{30,\,200\} = 30 \\
\end{align*}
\]

Or, equivalently, for each $i=1,2,3,4$:

\[
0 \leq x_i \leq \min\{\text{Demand}_i,\,\text{InitialInventory}_i\}
\]

All variables are continuous and nonnegative.

Complete numerical formulation:

\[
\begin{align*}
\max\quad & 119.144\,x_1 + 119.144\,x_2 + 120.144\,x_3 + 121.244\,x_4 \\
\text{s.t.}\quad
& 0 \leq x_1 \leq 30 \\
& 0 \leq x_2 \leq 40 \\
& 0 \leq x_3 \leq 50 \\
& 0 \leq x_4 \leq 30 \\
& x_1, x_2, x_3, x_4 \geq 0
\end{align*}
\]