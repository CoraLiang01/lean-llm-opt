Let $x_{sc}$ denote the amount of fresh produce shipped from supplier $s$ to customer $c$.

**Sets and Indices:**
- Suppliers $s \in \{\text{Supplier1}, \text{Supplier2}, \text{Supplier3}, \text{Supplier4}, \text{Supplier5}\}$
- Customers $c \in \{\text{Customer1}, \text{Customer2}, \text{Customer3}, \text{Customer4}, \text{Customer5}, \text{Customer6}\}$

**Parameters:**
- $d_c$: demand for customer $c$
    - $d_{\text{Customer1}} = 70$
    - $d_{\text{Customer2}} = 80$
    - $d_{\text{Customer3}} = 60$
    - $d_{\text{Customer4}} = 90$
    - $d_{\text{Customer5}} = 85$
    - $d_{\text{Customer6}} = 95$
- $u_s$: supply capacity of supplier $s$
    - $u_{\text{Supplier1}} = 200$
    - $u_{\text{Supplier2}} = 250$
    - $u_{\text{Supplier3}} = 230$
    - $u_{\text{Supplier4}} = 220$
    - $u_{\text{Supplier5}} = 210$
- $c_{sc}$: transportation cost per unit from supplier $s$ to customer $c$ (see table below)

|            | Customer1 | Customer2 | Customer3 | Customer4 | Customer5 | Customer6 |
|------------|-----------|-----------|-----------|-----------|-----------|-----------|
| Supplier1  |     2     |     3     |     1     |     2     |     3     |     2     |
| Supplier2  |     1     |     2     |     3     |     2     |     3     |     2     |
| Supplier3  |     3     |     1     |     2     |     3     |     2     |     3     |
| Supplier4  |     2     |     3     |     2     |     1     |     3     |     4     |
| Supplier5  |     3     |     2     |     3     |     3     |     2     |     3     |

**Decision Variables:**
- $x_{sc} \geq 0$ (continuous), for all $s, c$

---

### Objective

Minimize total transportation cost:
$$
\min \sum_{s} \sum_{c} c_{sc} \cdot x_{sc}
$$

That is,
\[
\min \Bigg[
\begin{aligned}
&2x_{\text{Supplier1, Customer1}} + 3x_{\text{Supplier1, Customer2}} + 1x_{\text{Supplier1, Customer3}} + 2x_{\text{Supplier1, Customer4}} + 3x_{\text{Supplier1, Customer5}} + 2x_{\text{Supplier1, Customer6}} \\
+&1x_{\text{Supplier2, Customer1}} + 2x_{\text{Supplier2, Customer2}} + 3x_{\text{Supplier2, Customer3}} + 2x_{\text{Supplier2, Customer4}} + 3x_{\text{Supplier2, Customer5}} + 2x_{\text{Supplier2, Customer6}} \\
+&3x_{\text{Supplier3, Customer1}} + 1x_{\text{Supplier3, Customer2}} + 2x_{\text{Supplier3, Customer3}} + 3x_{\text{Supplier3, Customer4}} + 2x_{\text{Supplier3, Customer5}} + 3x_{\text{Supplier3, Customer6}} \\
+&2x_{\text{Supplier4, Customer1}} + 3x_{\text{Supplier4, Customer2}} + 2x_{\text{Supplier4, Customer3}} + 1x_{\text{Supplier4, Customer4}} + 3x_{\text{Supplier4, Customer5}} + 4x_{\text{Supplier4, Customer6}} \\
+&3x_{\text{Supplier5, Customer1}} + 2x_{\text{Supplier5, Customer2}} + 3x_{\text{Supplier5, Customer3}} + 3x_{\text{Supplier5, Customer4}} + 2x_{\text{Supplier5, Customer5}} + 3x_{\text{Supplier5, Customer6}}
\end{aligned}
\Bigg]
\]

---

### Constraints

**1. Demand Satisfaction (for each customer):**
\[
\sum_{s} x_{sc} = d_c \qquad \forall c
\]
Explicitly:
\[
\begin{aligned}
x_{\text{Supplier1, Customer1}} + x_{\text{Supplier2, Customer1}} + x_{\text{Supplier3, Customer1}} + x_{\text{Supplier4, Customer1}} + x_{\text{Supplier5, Customer1}} &= 70 \\
x_{\text{Supplier1, Customer2}} + x_{\text{Supplier2, Customer2}} + x_{\text{Supplier3, Customer2}} + x_{\text{Supplier4, Customer2}} + x_{\text{Supplier5, Customer2}} &= 80 \\
x_{\text{Supplier1, Customer3}} + x_{\text{Supplier2, Customer3}} + x_{\text{Supplier3, Customer3}} + x_{\text{Supplier4, Customer3}} + x_{\text{Supplier5, Customer3}} &= 60 \\
x_{\text{Supplier1, Customer4}} + x_{\text{Supplier2, Customer4}} + x_{\text{Supplier3, Customer4}} + x_{\text{Supplier4, Customer4}} + x_{\text{Supplier5, Customer4}} &= 90 \\
x_{\text{Supplier1, Customer5}} + x_{\text{Supplier2, Customer5}} + x_{\text{Supplier3, Customer5}} + x_{\text{Supplier4, Customer5}} + x_{\text{Supplier5, Customer5}} &= 85 \\
x_{\text{Supplier1, Customer6}} + x_{\text{Supplier2, Customer6}} + x_{\text{Supplier3, Customer6}} + x_{\text{Supplier4, Customer6}} + x_{\text{Supplier5, Customer6}} &= 95 \\
\end{aligned}
\]

**2. Supply Capacity (for each supplier):**
\[
\sum_{c} x_{sc} \leq u_s \qquad \forall s
\]
Explicitly:
\[
\begin{aligned}
x_{\text{Supplier1, Customer1}} + x_{\text{Supplier1, Customer2}} + x_{\text{Supplier1, Customer3}} + x_{\text{Supplier1, Customer4}} + x_{\text{Supplier1, Customer5}} + x_{\text{Supplier1, Customer6}} &\leq 200 \\
x_{\text{Supplier2, Customer1}} + x_{\text{Supplier2, Customer2}} + x_{\text{Supplier2, Customer3}} + x_{\text{Supplier2, Customer4}} + x_{\text{Supplier2, Customer5}} + x_{\text{Supplier2, Customer6}} &\leq 250 \\
x_{\text{Supplier3, Customer1}} + x_{\text{Supplier3, Customer2}} + x_{\text{Supplier3, Customer3}} + x_{\text{Supplier3, Customer4}} + x_{\text{Supplier3, Customer5}} + x_{\text{Supplier3, Customer6}} &\leq 230 \\
x_{\text{Supplier4, Customer1}} + x_{\text{Supplier4, Customer2}} + x_{\text{Supplier4, Customer3}} + x_{\text{Supplier4, Customer4}} + x_{\text{Supplier4, Customer5}} + x_{\text{Supplier4, Customer6}} &\leq 220 \\
x_{\text{Supplier5, Customer1}} + x_{\text{Supplier5, Customer2}} + x_{\text{Supplier5, Customer3}} + x_{\text{Supplier5, Customer4}} + x_{\text{Supplier5, Customer5}} + x_{\text{Supplier5, Customer6}} &\leq 210 \\
\end{aligned}
\]

**3. Nonnegativity:**
\[
x_{sc} \geq 0 \qquad \forall s, c
\]

---

**Summary:**  
Minimize total transportation cost by choosing nonnegative shipment quantities $x_{sc}$ from each supplier to each customer, so that all customer demands are exactly met and no supplier exceeds its capacity, using the costs and limits as given above.