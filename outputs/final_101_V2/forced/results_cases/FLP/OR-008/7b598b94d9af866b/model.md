##### Decision Variables

Let $x_{ij} \geq 0$ denote the quantity of fresh produce shipped from warehouse (supplier) $i$ to store (customer) $j$, for all $i \in I$, $j \in J$.

##### Parameters

- $I = \{\text{Supplier1}, \text{Supplier2}, \text{Supplier3}, \text{Supplier4}, \text{Supplier5}\}$
- $J = \{\text{Customer1}, \text{Customer2}, \text{Customer3}, \text{Customer4}, \text{Customer5}, \text{Customer6}\}$

- Demand for each customer:
  - $d_{\text{Customer1}} = 70$
  - $d_{\text{Customer2}} = 80$
  - $d_{\text{Customer3}} = 60$
  - $d_{\text{Customer4}} = 90$
  - $d_{\text{Customer5}} = 85$
  - $d_{\text{Customer6}} = 95$

- Supply capacity for each supplier:
  - $s_{\text{Supplier1}} = 200$
  - $s_{\text{Supplier2}} = 250$
  - $s_{\text{Supplier3}} = 230$
  - $s_{\text{Supplier4}} = 220$
  - $s_{\text{Supplier5}} = 210$

- Transportation cost per unit ($c_{ij}$):

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

##### Constraints

1. **Demand satisfaction (for each customer):**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supply capacity (for each supplier):**
   \[
   \sum_{j \in J} x_{ij} \leq s_i, \quad \forall i \in I
   \]

3. **Nonnegativity:**
   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]

##### Explicit Model with Data

\[
\begin{align*}
\min\ & 
  2x_{\text{Supplier1},\text{Customer1}} + 3x_{\text{Supplier1},\text{Customer2}} + 1x_{\text{Supplier1},\text{Customer3}} + 2x_{\text{Supplier1},\text{Customer4}} + 3x_{\text{Supplier1},\text{Customer5}} + 2x_{\text{Supplier1},\text{Customer6}} \\
&+ 1x_{\text{Supplier2},\text{Customer1}} + 2x_{\text{Supplier2},\text{Customer2}} + 3x_{\text{Supplier2},\text{Customer3}} + 2x_{\text{Supplier2},\text{Customer4}} + 3x_{\text{Supplier2},\text{Customer5}} + 2x_{\text{Supplier2},\text{Customer6}} \\
&+ 3x_{\text{Supplier3},\text{Customer1}} + 1x_{\text{Supplier3},\text{Customer2}} + 2x_{\text{Supplier3},\text{Customer3}} + 3x_{\text{Supplier3},\text{Customer4}} + 2x_{\text{Supplier3},\text{Customer5}} + 3x_{\text{Supplier3},\text{Customer6}} \\
&+ 2x_{\text{Supplier4},\text{Customer1}} + 3x_{\text{Supplier4},\text{Customer2}} + 2x_{\text{Supplier4},\text{Customer3}} + 1x_{\text{Supplier4},\text{Customer4}} + 3x_{\text{Supplier4},\text{Customer5}} + 4x_{\text{Supplier4},\text{Customer6}} \\
&+ 3x_{\text{Supplier5},\text{Customer1}} + 2x_{\text{Supplier5},\text{Customer2}} + 3x_{\text{Supplier5},\text{Customer3}} + 3x_{\text{Supplier5},\text{Customer4}} + 2x_{\text{Supplier5},\text{Customer5}} + 3x_{\text{Supplier5},\text{Customer6}}
\end{align*}
\]

Subject to:

- $\sum_{i \in I} x_{i,\text{Customer1}} = 70$
- $\sum_{i \in I} x_{i,\text{Customer2}} = 80$
- $\sum_{i \in I} x_{i,\text{Customer3}} = 60$
- $\sum_{i \in I} x_{i,\text{Customer4}} = 90$
- $\sum_{i \in I} x_{i,\text{Customer5}} = 85$
- $\sum_{i \in I} x_{i,\text{Customer6}} = 95$

- $\sum_{j \in J} x_{\text{Supplier1},j} \leq 200$
- $\sum_{j \in J} x_{\text{Supplier2},j} \leq 250$
- $\sum_{j \in J} x_{\text{Supplier3},j} \leq 230$
- $\sum_{j \in J} x_{\text{Supplier4},j} \leq 220$
- $\sum_{j \in J} x_{\text{Supplier5},j} \leq 210$

- $x_{ij} \geq 0$ for all $i \in I$, $j \in J$.

---

**All parameters, vectors, and matrices are explicitly included as retrieved from the CSV files.**