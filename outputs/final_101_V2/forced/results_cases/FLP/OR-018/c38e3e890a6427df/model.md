##### Decision Variables

$x_i \geq 0$: Number of units of Baby product $i$ to fulfill (continuous).

##### Parameters

- $I = \{\text{Baby Food\_255.28}\}$
- Revenue per unit: $r_i$  
  $r_{\text{Baby Food\_255.28}} = 255.28$
- Initial inventory: $s_i$  
  $s_{\text{Baby Food\_255.28}} = 22,\!749,\!210$
- Demand: $d_i$  
  $d_{\text{Baby Food\_255.28}} = 3,\!066,\!513$

##### Objective Function

\[
\max \sum_{i \in I} r_i x_i
\]
That is,
\[
\max\ 255.28\, x_{\text{Baby Food\_255.28}}
\]

##### Constraints

1. Inventory constraint: $x_i \leq s_i,\quad \forall i \in I$  
   $x_{\text{Baby Food\_255.28}} \leq 22,\!749,\!210$
2. Demand constraint: $x_i \leq d_i,\quad \forall i \in I$  
   $x_{\text{Baby Food\_255.28}} \leq 3,\!066,\!513$
3. Nonnegativity: $x_i \geq 0,\quad \forall i \in I$

##### Complete Model

\[
\begin{align*}
\max\quad & 255.28\, x_{\text{Baby Food\_255.28}} \\
\text{s.t.}\quad & x_{\text{Baby Food\_255.28}} \leq 22,\!749,\!210 \\
                & x_{\text{Baby Food\_255.28}} \leq 3,\!066,\!513 \\
                & x_{\text{Baby Food\_255.28}} \geq 0
\end{align*}
\]

Where $x_{\text{Baby Food\_255.28}}$ is the number of units of "Baby Food_255.28" to fulfill, bounded by both initial inventory and demand.