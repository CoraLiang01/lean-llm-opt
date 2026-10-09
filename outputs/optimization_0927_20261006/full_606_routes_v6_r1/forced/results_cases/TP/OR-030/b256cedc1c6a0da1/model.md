Let $I$ be the set of all car models classified under ‘FDK57’, indexed in source order as $i=1,\ldots,6$.

#### Parameters (from retrieved data, in source order)
\[
\begin{array}{cccc}
\text{Index } i & \text{Revenue}_i & \text{Demand}_i & \text{Initial Inventory}_i \\
1 & 119.144 & 30 & 200 \\
2 & 119.144 & 40 & 100 \\
3 & 120.144 & 50 & 150 \\
4 & 121.244 & 30 & 200 \\
5 & 120.544 & 10 & 150 \\
6 & 120.844 & 50 & 150 \\
\end{array}
\]

#### Decision Variables
\[
x_i \geq 0 \quad \text{(continuous)},\quad \forall i=1,\ldots,6
\]
where $x_i$ is the quantity of car model $i$ to fulfill.

#### Objective Function
\[
\max \sum_{i=1}^6 \text{Revenue}_i \cdot x_i
= 119.144\,x_1 + 119.144\,x_2 + 120.144\,x_3 + 121.244\,x_4 + 120.544\,x_5 + 120.844\,x_6
\]

#### Constraints

1. Inventory and demand bounds for each model:
   \[
   0 \leq x_i \leq \min\{\text{Demand}_i,\,\text{Initial Inventory}_i\},\quad \forall i=1,\ldots,6
   \]
   That is:
   \begin{align*}
   0 \leq x_1 \leq 30 \\
   0 \leq x_2 \leq 40 \\
   0 \leq x_3 \leq 50 \\
   0 \leq x_4 \leq 30 \\
   0 \leq x_5 \leq 10 \\
   0 \leq x_6 \leq 50 \\
   \end{align*}

#### Complete Model

\[
\begin{align*}
\max\quad & 119.144\,x_1 + 119.144\,x_2 + 120.144\,x_3 + 121.244\,x_4 + 120.544\,x_5 + 120.844\,x_6 \\
\text{s.t.}\quad
& 0 \leq x_1 \leq 30 \\
& 0 \leq x_2 \leq 40 \\
& 0 \leq x_3 \leq 50 \\
& 0 \leq x_4 \leq 30 \\
& 0 \leq x_5 \leq 10 \\
& 0 \leq x_6 \leq 50 \\
& x_i \geq 0,\quad i=1,\ldots,6
\end{align*}
\]

All variables are continuous and nonnegative. The model maximizes total revenue from fulfilling demand for each FDK57 car model, subject to inventory and demand limits.