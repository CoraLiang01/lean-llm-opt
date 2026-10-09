##### Sets and Parameters

Let $I$ be the set of products classified as ‘ELE-S’:
\[
I = \{\text{ELE-SMA-10000463},\ \text{ELE-SMA-10000487},\ \text{ELE-SMA-10003333},\ \text{ELE-SMA-10009012},\ \text{ELE-SMA-10009999},\ \text{ELE-SMA-10011234},\ \text{ELE-SMA-10027456},\ \text{ELE-SMA-10028567}\}
\]

For each product $i \in I$:

\[
\begin{array}{llll}
\text{Product Reference} & \text{Revenue } (r_i) & \text{Initial Inventory } (s_i) & \text{Demand } (d_i) \\
\hline
\text{ELE-SMA-10000463} & 4.0 & 2000.0 & 295 \\
\text{ELE-SMA-10000487} & 14.0 & 7000.0 & 1002 \\
\text{ELE-SMA-10003333} & 14.0 & 7000.0 & 958 \\
\text{ELE-SMA-10009012} & 4.0 & 6000.0 & 777 \\
\text{ELE-SMA-10009999} & 4.0 & 2000.0 & 271 \\
\text{ELE-SMA-10011234} & 4.0 & 2000.0 & 244 \\
\text{ELE-SMA-10027456} & 14.0 & 7000.0 & 990 \\
\text{ELE-SMA-10028567} & 14.0 & 7000.0 & 1000 \\
\end{array}
\]

##### Decision Variables

For each $i \in I$:

\[
x_i \geq 0
\]
Number of units of product $i$ to fulfill.

##### Objective Function

\[
\max \sum_{i \in I} r_i x_i
\]

##### Constraints

1. Inventory and demand limits for each product:
   \[
   0 \leq x_i \leq \min\{s_i, d_i\}, \quad \forall i \in I
   \]

##### Complete Model

\[
\begin{align*}
\max\ & \sum_{i \in I} r_i x_i \\
\text{s.t.}\quad & 0 \leq x_i \leq \min\{s_i, d_i\}, \quad \forall i \in I \\
& x_i \geq 0, \quad \forall i \in I
\end{align*}
\]

Where the parameters are:

\[
\begin{array}{llll}
i & r_i & s_i & d_i \\
\hline
\text{ELE-SMA-10000463} & 4.0 & 2000.0 & 295 \\
\text{ELE-SMA-10000487} & 14.0 & 7000.0 & 1002 \\
\text{ELE-SMA-10003333} & 14.0 & 7000.0 & 958 \\
\text{ELE-SMA-10009012} & 4.0 & 6000.0 & 777 \\
\text{ELE-SMA-10009999} & 4.0 & 2000.0 & 271 \\
\text{ELE-SMA-10011234} & 4.0 & 2000.0 & 244 \\
\text{ELE-SMA-10027456} & 14.0 & 7000.0 & 990 \\
\text{ELE-SMA-10028567} & 14.0 & 7000.0 & 1000 \\
\end{array}
\]

All variables $x_i$ are continuous and nonnegative.