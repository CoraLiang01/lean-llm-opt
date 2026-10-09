**Sets:**  
Let $I = \{\text{Supplier1}, \text{Supplier2}, \text{Supplier3}, \text{Supplier4}, \text{Supplier5}\}$ (warehouses)  
Let $J = \{\text{Customer1}, \text{Customer2}, \text{Customer3}, \text{Customer4}, \text{Customer5}, \text{Customer6}\}$ (stores)

**Parameters:**  
Demands:  
\[
\begin{align*}
d_{\text{Customer1}} &= 70 \\
d_{\text{Customer2}} &= 80 \\
d_{\text{Customer3}} &= 60 \\
d_{\text{Customer4}} &= 90 \\
d_{\text{Customer5}} &= 85 \\
d_{\text{Customer6}} &= 95 \\
\end{align*}
\]

Supply capacities:  
\[
\begin{align*}
s_{\text{Supplier1}} &= 200 \\
s_{\text{Supplier2}} &= 250 \\
s_{\text{Supplier3}} &= 230 \\
s_{\text{Supplier4}} &= 220 \\
s_{\text{Supplier5}} &= 210 \\
\end{align*}
\]

Transportation costs $c_{ij}$ (from supplier $i$ to customer $j$):

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

**Decision Variables:**  
For each $i\in I$, $j\in J$:  
$x_{ij} \geq 0$ (continuous): quantity shipped from supplier $i$ to customer $j$

**Objective:**  
\[
\min \sum_{i\in I} \sum_{j\in J} c_{ij} x_{ij}
\]

**Constraints:**

1. **Demand satisfaction:**  
For each $j\in J$,
\[
\sum_{i\in I} x_{ij} \geq d_j
\]
i.e.,
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

2. **Supply capacity:**  
For each $i\in I$,
\[
\sum_{j\in J} x_{ij} \leq s_i
\]
i.e.,
\[
\begin{align*}
x_{\text{Supplier1},\text{Customer1}} + x_{\text{Supplier1},\text{Customer2}} + x_{\text{Supplier1},\text{Customer3}} + x_{\text{Supplier1},\text{Customer4}} + x_{\text{Supplier1},\text{Customer5}} + x_{\text{Supplier1},\text{Customer6}} &\leq 200 \\
x_{\text{Supplier2},\text{Customer1}} + x_{\text{Supplier2},\text{Customer2}} + x_{\text{Supplier2},\text{Customer3}} + x_{\text{Supplier2},\text{Customer4}} + x_{\text{Supplier2},\text{Customer5}} + x_{\text{Supplier2},\text{Customer6}} &\leq 250 \\
x_{\text{Supplier3},\text{Customer1}} + x_{\text{Supplier3},\text{Customer2}} + x_{\text{Supplier3},\text{Customer3}} + x_{\text{Supplier3},\text{Customer4}} + x_{\text{Supplier3},\text{Customer5}} + x_{\text{Supplier3},\text{Customer6}} &\leq 230 \\
x_{\text{Supplier4},\text{Customer1}} + x_{\text{Supplier4},\text{Customer2}} + x_{\text{Supplier4},\text{Customer3}} + x_{\text{Supplier4},\text{Customer4}} + x_{\text{Supplier4},\text{Customer5}} + x_{\text{Supplier4},\text{Customer6}} &\leq 220 \\
x_{\text{Supplier5},\text{Customer1}} + x_{\text{Supplier5},\text{Customer2}} + x_{\text{Supplier5},\text{Customer3}} + x_{\text{Supplier5},\text{Customer4}} + x_{\text{Supplier5},\text{Customer5}} + x_{\text{Supplier5},\text{Customer6}} &\leq 210 \\
\end{align*}
\]

3. **Nonnegativity:**  
\[
x_{ij} \geq 0 \quad \forall i\in I,\, j\in J
\]

**Full Model:**

\[
\begin{align*}
\min\ & \sum_{i\in I} \sum_{j\in J} c_{ij} x_{ij} \\
\text{s.t.}\quad
& \sum_{i\in I} x_{ij} \geq d_j \quad \forall j\in J \\
& \sum_{j\in J} x_{ij} \leq s_i \quad \forall i\in I \\
& x_{ij} \geq 0 \quad \forall i\in I,\, j\in J \\
\end{align*}
\]
with all parameters and indices as specified above.