##### Decision Variables

$x_i \geq 0$: quantity of ‘FDK57’ car model $i$ to fulfill, for $i = 1, \ldots, 6$ (continuous).

##### Parameters

Let $I = \{1,2,3,4,5,6\}$ index the six ‘FDK57’ car model records.

- Revenue per unit: $r = [119.144,\ 120.144,\ 121.244,\ 119.144,\ 120.144,\ 121.244]$
- Initial Inventory: $s = [200,\ 150,\ 150,\ 200,\ 150,\ 150]$
- Demand: $d = [30,\ 50,\ 10,\ 30,\ 50,\ 10]$

##### Objective Function

\[
\max \sum_{i=1}^6 r_i x_i
\]
where $r_i$ is the revenue per unit for model $i$.

##### Constraints

1. Inventory limit: $x_i \leq s_i,\quad \forall i \in I$
2. Demand limit: $x_i \leq d_i,\quad \forall i \in I$
3. Nonnegativity: $x_i \geq 0,\quad \forall i \in I$

##### Full Model

\[
\begin{align*}
\max\quad & 119.144\,x_1 + 120.144\,x_2 + 121.244\,x_3 + 119.144\,x_4 + 120.144\,x_5 + 121.244\,x_6 \\
\text{s.t.}\quad & x_1 \leq 200 \\
& x_2 \leq 150 \\
& x_3 \leq 150 \\
& x_4 \leq 200 \\
& x_5 \leq 150 \\
& x_6 \leq 150 \\
& x_1 \leq 30 \\
& x_2 \leq 50 \\
& x_3 \leq 10 \\
& x_4 \leq 30 \\
& x_5 \leq 50 \\
& x_6 \leq 10 \\
& x_i \geq 0,\quad i=1,\ldots,6
\end{align*}
\]

##### Retrieved Information

- Model indices: $I = \{1,2,3,4,5,6\}$
- Revenue: $[119.144,\ 120.144,\ 121.244,\ 119.144,\ 120.144,\ 121.244]$
- Initial Inventory: $[200,\ 150,\ 150,\ 200,\ 150,\ 150]$
- Demand: $[30,\ 50,\ 10,\ 30,\ 50,\ 10]$