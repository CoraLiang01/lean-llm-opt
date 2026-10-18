Let $x_{ij}$ denote the amount of fresh produce shipped from supplier $i$ to customer $j$.

**Sets:**
- Suppliers $i \in \{\text{Supplier1}, \text{Supplier2}, \text{Supplier3}, \text{Supplier4}, \text{Supplier5}\}$
- Customers $j \in \{\text{Customer1}, \text{Customer2}, \text{Customer3}, \text{Customer4}, \text{Customer5}, \text{Customer6}\}$

**Parameters:**
- Demand for each customer:
  - $\text{Customer1}: 70$
  - $\text{Customer2}: 80$
  - $\text{Customer3}: 60$
  - $\text{Customer4}: 90$
  - $\text{Customer5}: 85$
  - $\text{Customer6}: 95$
- Supply capacity for each supplier:
  - $\text{Supplier1}: 200$
  - $\text{Supplier2}: 250$
  - $\text{Supplier3}: 230$
  - $\text{Supplier4}: 220$
  - $\text{Supplier5}: 210$
- Transportation costs $c_{ij}$ (per unit from supplier $i$ to customer $j$):

|            | Customer1 | Customer2 | Customer3 | Customer4 | Customer5 | Customer6 |
|------------|-----------|-----------|-----------|-----------|-----------|-----------|
| Supplier1  |     2     |     3     |     1     |     2     |     3     |     2     |
| Supplier2  |     1     |     2     |     3     |     2     |     3     |     2     |
| Supplier3  |     3     |     1     |     2     |     3     |     2     |     3     |
| Supplier4  |     2     |     3     |     2     |     1     |     3     |     4     |
| Supplier5  |     3     |     2     |     3     |     3     |     2     |     3     |

**Decision Variables:**
- $x_{ij} \geq 0$ (continuous), for all suppliers $i$ and customers $j$

---

### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in \text{Suppliers}} \sum_{j \in \text{Customers}} c_{ij} x_{ij}
$$

Explicitly:
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

---

### Constraints

**1. Demand satisfaction (for each customer):**
\[
\sum_{i \in \text{Suppliers}} x_{ij} = d_j, \quad \forall j \in \text{Customers}
\]
Explicitly:
\[
\begin{align*}
x_{\text{Supplier1},\text{Customer1}} + x_{\text{Supplier2},\text{Customer1}} + x_{\text{Supplier3},\text{Customer1}} + x_{\text{Supplier4},\text{Customer1}} + x_{\text{Supplier5},\text{Customer1}} &= 70 \\
x_{\text{Supplier1},\text{Customer2}} + x_{\text{Supplier2},\text{Customer2}} + x_{\text{Supplier3},\text{Customer2}} + x_{\text{Supplier4},\text{Customer2}} + x_{\text{Supplier5},\text{Customer2}} &= 80 \\
x_{\text{Supplier1},\text{Customer3}} + x_{\text{Supplier2},\text{Customer3}} + x_{\text{Supplier3},\text{Customer3}} + x_{\text{Supplier4},\text{Customer3}} + x_{\text{Supplier5},\text{Customer3}} &= 60 \\
x_{\text{Supplier1},\text{Customer4}} + x_{\text{Supplier2},\text{Customer4}} + x_{\text{Supplier3},\text{Customer4}} + x_{\text{Supplier4},\text{Customer4}} + x_{\text{Supplier5},\text{Customer4}} &= 90 \\
x_{\text{Supplier1},\text{Customer5}} + x_{\text{Supplier2},\text{Customer5}} + x_{\text{Supplier3},\text{Customer5}} + x_{\text{Supplier4},\text{Customer5}} + x_{\text{Supplier5},\text{Customer5}} &= 85 \\
x_{\text{Supplier1},\text{Customer6}} + x_{\text{Supplier2},\text{Customer6}} + x_{\text{Supplier3},\text{Customer6}} + x_{\text{Supplier4},\text{Customer6}} + x_{\text{Supplier5},\text{Customer6}} &= 95 \\
\end{align*}
\]

**2. Supply capacity (for each supplier):**
\[
\sum_{j \in \text{Customers}} x_{ij} \leq s_i, \quad \forall i \in \text{Suppliers}
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

**3. Nonnegativity:**
\[
x_{ij} \geq 0 \quad \forall i, j
\]

---

**Summary:**  
Minimize total transportation cost by choosing nonnegative shipment quantities $x_{ij}$ from each supplier to each customer, so that all customer demands are exactly met and no supplier exceeds its capacity, using the costs and limits as given above.