##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity of fresh produce shipped from warehouse (supplier) $i$ to store (customer) $j$, for all $i \in I$, $j \in J$.

Where:
- $I = \{\text{Supplier1}, \text{Supplier2}, \text{Supplier3}, \text{Supplier4}, \text{Supplier5}\}$
- $J = \{\text{Customer1}, \text{Customer2}, \text{Customer3}, \text{Customer4}, \text{Customer5}, \text{Customer6}\}$

##### Parameters

- Demands:
  - $d_{\text{Customer1}} = 70$
  - $d_{\text{Customer2}} = 80$
  - $d_{\text{Customer3}} = 60$
  - $d_{\text{Customer4}} = 90$
  - $d_{\text{Customer5}} = 85$
  - $d_{\text{Customer6}} = 95$

- Supply capacities:
  - $s_{\text{Supplier1}} = 200$
  - $s_{\text{Supplier2}} = 250$
  - $s_{\text{Supplier3}} = 230$
  - $s_{\text{Supplier4}} = 220$
  - $s_{\text{Supplier5}} = 210$

- Transportation costs $c_{ij}$ (from transportation_costs.csv):

\[
\begin{array}{c|cccccc}
 & \text{Customer1} & \text{Customer2} & \text{Customer3} & \text{Customer4} & \text{Customer5} & \text{Customer6} \\
\hline
\text{Supplier1} & 2 & 3 & 1 & 2 & 3 & 2 \\
\text{Supplier2} & 1 & 2 & 3 & 2 & 3 & 2 \\
\text{Supplier3} & 3 & 1 & 2 & 3 & 2 & 3 \\
\text{Supplier4} & 2 & 3 & 2 & 1 & 3 & 4 \\
\text{Supplier5} & 3 & 2 & 3 & 3 & 2 & 3 \\
\end{array}
\]

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

That is,

\[
\begin{align*}
\min\ & 2x_{\text{Supplier1},\text{Customer1}} + 3x_{\text{Supplier1},\text{Customer2}} + 1x_{\text{Supplier1},\text{Customer3}} + 2x_{\text{Supplier1},\text{Customer4}} + 3x_{\text{Supplier1},\text{Customer5}} + 2x_{\text{Supplier1},\text{Customer6}} \\
&+ 1x_{\text{Supplier2},\text{Customer1}} + 2x_{\text{Supplier2},\text{Customer2}} + 3x_{\text{Supplier2},\text{Customer3}} + 2x_{\text{Supplier2},\text{Customer4}} + 3x_{\text{Supplier2},\text{Customer5}} + 2x_{\text{Supplier2},\text{Customer6}} \\
&+ 3x_{\text{Supplier3},\text{Customer1}} + 1x_{\text{Supplier3},\text{Customer2}} + 2x_{\text{Supplier3},\text{Customer3}} + 3x_{\text{Supplier3},\text{Customer4}} + 2x_{\text{Supplier3},\text{Customer5}} + 3x_{\text{Supplier3},\text{Customer6}} \\
&+ 2x_{\text{Supplier4},\text{Customer1}} + 3x_{\text{Supplier4},\text{Customer2}} + 2x_{\text{Supplier4},\text{Customer3}} + 1x_{\text{Supplier4},\text{Customer4}} + 3x_{\text{Supplier4},\text{Customer5}} + 4x_{\text{Supplier4},\text{Customer6}} \\
&+ 3x_{\text{Supplier5},\text{Customer1}} + 2x_{\text{Supplier5},\text{Customer2}} + 3x_{\text{Supplier5},\text{Customer3}} + 3x_{\text{Supplier5},\text{Customer4}} + 2x_{\text{Supplier5},\text{Customer5}} + 3x_{\text{Supplier5},\text{Customer6}}
\end{align*}
\]

##### Constraints

1. **Demand satisfaction (each customer receives at least its demand):**

For each $j \in J$,

\[
\sum_{i \in I} x_{ij} \geq d_j
\]

Explicitly:

\[
\begin{align*}
x_{\text{Supplier1},\text{Customer1}} + x_{\text{Supplier2},\text{Customer1}} + x_{\text{Supplier3},\text{Customer1}} + x_{\text{Supplier4},\text{Customer1}} + x_{\text{Supplier5},\text{Customer1}} &\geq 70 \\
x_{\text{Supplier1},\text{Customer2}} + x_{\text{Supplier2},\text{Customer2}} + x_{\text{Supplier3},\text{Customer2}} + x_{\text{Supplier4},\text{Customer2}} + x_{\text{Supplier5},\text{Customer2}} &\geq 80 \\
x_{\text{Supplier1},\text{Customer3}} + x_{\text{Supplier2},\text{Customer3}} + x_{\text{Supplier3},\text{Customer3}} + x_{\text{Supplier4},\text{Customer3}} + x_{\text{Supplier5},\text{Customer3}} &\geq 60 \\
x_{\text{Supplier1},\text{Customer4}} + x_{\text{Supplier2},\text{Customer4}} + x_{\text{Supplier3},\text{Customer4}} + x_{\text{Supplier4},\text{Customer4}} + x_{\text{Supplier5},\text{Customer4}} &\geq 90 \\
x_{\text{Supplier1},\text{Customer5}} + x_{\text{Supplier2},\text{Customer5}} + x_{\text{Supplier3},\text{Customer5}} + x_{\text{Supplier4},\text{Customer5}} + x_{\text{Supplier5},\text{Customer5}} &\geq 85 \\
x_{\text{Supplier1},\text{Customer6}} + x_{\text{Supplier2},\text{Customer6}} + x_{\text{Supplier3},\text{Customer6}} + x_{\text{Supplier4},\text{Customer6}} + x_{\text{Supplier5},\text{Customer6}} &\geq 95 \\
\end{align*}
\]

2. **Supply capacity (each supplier cannot ship more than its capacity):**

For each $i \in I$,

\[
\sum_{j \in J} x_{ij} \leq s_i
\]

Explicitly:

\[
\begin{align*}
x_{\text{Supplier1},\text{Customer1}} + x_{\text{Supplier1},\text{Customer2}} + x_{\text{Supplier1},\text{Customer3}} + x_{\text{Supplier1},\text{Customer4}} + x_{\text{Supplier1},\text{Customer5}} + x_{\text{Supplier1},\text{Customer6}} &\leq 200 \\
x_{\text{Supplier2},\text{Customer1}} + x_{\text{Supplier2},\text{Customer2}} + x_{\text{Supplier2},\text{Customer3}} + x_{\text{Supplier2},\text{Customer4}} + x_{\text{Supplier2},\text{Customer5}} + x_{\text{Supplier2},\text{Customer6}} &\leq 250 \\
x_{\text{Supplier3},\text{Customer1}} + x_{\text{Supplier3},\text{Customer2}} + x_{\text{Supplier3},\text{Customer3}} + x_{\text{Supplier3},\text{Customer4}} + x_{\text{Supplier3},\text{Customer5}} + x_{\text{Supplier3},\text{Customer6}} &\leq 230 \\
x_{\text{Supplier4},\text{Customer1}} + x_{\text{Supplier4},\text{Customer2}} + x_{\text{Supplier4},\text{Customer3}} + x_{\text{Supplier4},\text{Customer4}} + x_{\text{Supplier4},\text{Customer5}} + x_{\text{Supplier4},\text{Customer6}} &\leq 220 \\
x_{\text{Supplier5},\text{Customer1}} + x_{\text{Supplier5},\text{Customer2}} + x_{\text{Supplier5},\text{Customer3}} + x_{\text{Supplier5},\text{Customer4}} + x_{\text{Supplier5},\text{Customer5}} + x_{\text{Supplier5},\text{Customer6}} &\leq 210 \\
\end{align*}
\]

3. **Non-negativity:**

\[
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\]

##### Complete Model

\[
\begin{align*}
\min\ & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{s.t.}\quad
& \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I \\
& x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\end{align*}
\]

with all parameters and coefficients as specified above.