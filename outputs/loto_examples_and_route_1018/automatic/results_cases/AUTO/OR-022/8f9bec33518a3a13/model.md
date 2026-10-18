Let $x_i$ denote the number of units of product $i$ (where $i$ indexes the ‘27in’ products) to be fulfilled.

Parameters (from the retrieved data):

\[
\begin{array}{llll}
\text{Product} & \text{Revenue}_i & \text{Demand}_i & \text{InitialInventory}_i \\
\hline
\text{27in 4K Gaming Monitor} & 261.2933 & 12474 & 62440 \\
\text{27in FHD Monitor} & 52.4965 & 15057 & 75500 \\
\end{array}
\]

Mathematical Model:

Objective:
\[
\max \; 261.2933\, x_1 + 52.4965\, x_2
\]

Subject to:
\[
\begin{align*}
& 0 \leq x_1 \leq \min\{12474,\, 62440\} \\
& 0 \leq x_2 \leq \min\{15057,\, 75500\} \\
& x_1,\, x_2 \in \mathbb{Z}_{\geq 0}
\end{align*}
\]

Where:
- $x_1$ = units of 27in 4K Gaming Monitor fulfilled
- $x_2$ = units of 27in FHD Monitor fulfilled

Or, explicitly:

\[
\begin{align*}
& 0 \leq x_1 \leq 12474 \\
& 0 \leq x_2 \leq 15057 \\
& x_1,\, x_2 \in \mathbb{Z}_{\geq 0}
\end{align*}
\]

Maximize total revenue from fulfilling ‘27in’ product demand, subject to inventory and demand limits.