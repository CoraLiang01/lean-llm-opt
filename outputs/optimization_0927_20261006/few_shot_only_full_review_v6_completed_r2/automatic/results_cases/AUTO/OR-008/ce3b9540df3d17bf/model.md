**Sets:**  
Let $I = \{\text{Supplier1}, \text{Supplier2}, \text{Supplier3}, \text{Supplier4}, \text{Supplier5}\}$ (warehouses)  
Let $J = \{\text{Customer1}, \text{Customer2}, \text{Customer3}, \text{Customer4}, \text{Customer5}, \text{Customer6}\}$ (stores)

**Parameters:**  
Demands:
- $d_{\text{Customer1}} = 70$
- $d_{\text{Customer2}} = 80$
- $d_{\text{Customer3}} = 60$
- $d_{\text{Customer4}} = 90$
- $d_{\text{Customer5}} = 85$
- $d_{\text{Customer6}} = 95$

Supply capacities:
- $s_{\text{Supplier1}} = 200$
- $s_{\text{Supplier2}} = 250$
- $s_{\text{Supplier3}} = 230$
- $s_{\text{Supplier4}} = 220$
- $s_{\text{Supplier5}} = 210$

Transportation costs $c_{ij}$ (per unit from supplier $i$ to customer $j$):

|            | Customer1 | Customer2 | Customer3 | Customer4 | Customer5 | Customer6 |
|------------|-----------|-----------|-----------|-----------|-----------|-----------|
| Supplier1  |     2     |     3     |     1     |     2     |     3     |     2     |
| Supplier2  |     1     |     2     |     3     |     2     |     3     |     2     |
| Supplier3  |     3     |     1     |     2     |     3     |     2     |     3     |
| Supplier4  |     2     |     3     |     2     |     1     |     3     |     4     |
| Supplier5  |     3     |     2     |     3     |     3     |     2     |     3     |

**Decision Variables:**  
For all $i \in I$, $j \in J$:
- $x_{ij} \geq 0$ (continuous): quantity shipped from supplier $i$ to customer $j$

**Objective:**  
Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$
That is,
\[
\min \Bigg[
\begin{aligned}
&2x_{\text{Supplier1},\text{Customer1}} + 3x_{\text{Supplier1},\text{Customer2}} + 1x_{\text{Supplier1},\text{Customer3}} + 2x_{\text{Supplier1},\text{Customer4}} + 3x_{\text{Supplier1},\text{Customer5}} + 2x_{\text{Supplier1},\text{Customer6}} \\
+&1x_{\text{Supplier2},\text{Customer1}} + 2x_{\text{Supplier2},\text{Customer2}} + 3x_{\text{Supplier2},\text{Customer3}} + 2x_{\text{Supplier2},\text{Customer4}} + 3x_{\text{Supplier2},\text{Customer5}} + 2x_{\text{Supplier2},\text{Customer6}} \\
+&3x_{\text{Supplier3},\text{Customer1}} + 1x_{\text{Supplier3},\text{Customer2}} + 2x_{\text{Supplier3},\text{Customer3}} + 3x_{\text{Supplier3},\text{Customer4}} + 2x_{\text{Supplier3},\text{Customer5}} + 3x_{\text{Supplier3},\text{Customer6}} \\
+&2x_{\text{Supplier4},\text{Customer1}} + 3x_{\text{Supplier4},\text{Customer2}} + 2x_{\text{Supplier4},\text{Customer3}} + 1x_{\text{Supplier4},\text{Customer4}} + 3x_{\text{Supplier4},\text{Customer5}} + 4x_{\text{Supplier4},\text{Customer6}} \\
+&3x_{\text{Supplier5},\text{Customer1}} + 2x_{\text{Supplier5},\text{Customer2}} + 3x_{\text{Supplier5},\text{Customer3}} + 3x_{\text{Supplier5},\text{Customer4}} + 2x_{\text{Supplier5},\text{Customer5}} + 3x_{\text{Supplier5},\text{Customer6}}
\Bigg]
\]

**Subject to:**

1. **Demand satisfaction (for each customer $j$):**
   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]
   Explicitly:
   \begin{align*}
   x_{\text{Supplier1},\text{Customer1}} + x_{\text{Supplier2},\text{Customer1}} + x_{\text{Supplier3},\text{Customer1}} + x_{\text{Supplier4},\text{Customer1}} + x_{\text{Supplier5},\text{Customer1}} &\geq 70 \\
   x_{\text{Supplier1},\text{Customer2}} + x_{\text{Supplier2},\text{Customer2}} + x_{\text{Supplier3},\text{Customer2}} + x_{\text{Supplier4},\text{Customer2}} + x_{\text{Supplier5},\text{Customer2}} &\geq 80 \\
   x_{\text{Supplier1},\text{Customer3}} + x_{\text{Supplier2},\text{Customer3}} + x_{\text{Supplier3},\text{Customer3}} + x_{\text{Supplier4},\text{Customer3}} + x_{\text{Supplier5},\text{Customer3}} &\geq 60 \\
   x_{\text{Supplier1},\text{Customer4}} + x_{\text{Supplier2},\text{Customer4}} + x_{\text{Supplier3},\text{Customer4}} + x_{\text{Supplier4},\text{Customer4}} + x_{\text{Supplier5},\text{Customer4}} &\geq 90 \\
   x_{\text{Supplier1},\text{Customer5}} + x_{\text{Supplier2},\text{Customer5}} + x_{\text{Supplier3},\text{Customer5}} + x_{\text{Supplier4},\text{Customer5}} + x_{\text{Supplier5},\text{Customer5}} &\geq 85 \\
   x_{\text{Supplier1},\text{Customer6}} + x_{\text{Supplier2},\text{Customer6}} + x_{\text{Supplier3},\text{Customer6}} + x_{\text{Supplier4},\text{Customer6}} + x_{\text{Supplier5},\text{Customer6}} &\geq 95 \\
   \end{align*}

2. **Supply capacity (for each supplier $i$):**
   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]
   Explicitly:
   \begin{align*}
   x_{\text{Supplier1},\text{Customer1}} + x_{\text{Supplier1},\text{Customer2}} + x_{\text{Supplier1},\text{Customer3}} + x_{\text{Supplier1},\text{Customer4}} + x_{\text{Supplier1},\text{Customer5}} + x_{\text{Supplier1},\text{Customer6}} &\leq 200 \\
   x_{\text{Supplier2},\text{Customer1}} + x_{\text{Supplier2},\text{Customer2}} + x_{\text{Supplier2},\text{Customer3}} + x_{\text{Supplier2},\text{Customer4}} + x_{\text{Supplier2},\text{Customer5}} + x_{\text{Supplier2},\text{Customer6}} &\leq 250 \\
   x_{\text{Supplier3},\text{Customer1}} + x_{\text{Supplier3},\text{Customer2}} + x_{\text{Supplier3},\text{Customer3}} + x_{\text{Supplier3},\text{Customer4}} + x_{\text{Supplier3},\text{Customer5}} + x_{\text{Supplier3},\text{Customer6}} &\leq 230 \\
   x_{\text{Supplier4},\text{Customer1}} + x_{\text{Supplier4},\text{Customer2}} + x_{\text{Supplier4},\text{Customer3}} + x_{\text{Supplier4},\text{Customer4}} + x_{\text{Supplier4},\text{Customer5}} + x_{\text{Supplier4},\text{Customer6}} &\leq 220 \\
   x_{\text{Supplier5},\text{Customer1}} + x_{\text{Supplier5},\text{Customer2}} + x_{\text{Supplier5},\text{Customer3}} + x_{\text{Supplier5},\text{Customer4}} + x_{\text{Supplier5},\text{Customer5}} + x_{\text{Supplier5},\text{Customer6}} &\leq 210 \\
   \end{align*}

3. **Non-negativity:**
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

**All identifiers and coefficients are preserved exactly as in the source data.**