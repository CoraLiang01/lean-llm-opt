##### Sets and Indices

- Let $F = \{\text{S1}, \text{S2}, \text{S3}\}$ be the set of warehouses, indexed by $i$.
- Let $C = \{\text{C1}, \text{C2}, \text{C3}\}$ be the set of musicians/bands (customers), indexed by $j$.

##### Parameters

- $f_i$: Fixed cost of opening warehouse $i$.
    - $f_{\text{S1}} = 102.33$
    - $f_{\text{S2}} = 94.92$
    - $f_{\text{S3}} = 91.83$
- $t_{ij}$: Transportation cost per unit from warehouse $i$ to customer $j$.

    |        | C1      | C2      | C3      |
    |--------|---------|---------|---------|
    | S1     | 1506.22 | 70.90   | 8.44    |
    | S2     | 1732.65 | 1780.72 | 567.44  |
    | S3     | 115.66  | 100.76  | 64.68   |

- $d_j$: Demand of customer $j$.
    - $d_{\text{C1}} = 1083$
    - $d_{\text{C2}} = 776$
    - $d_{\text{C3}} = 16214$

##### Decision Variables

- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise.
- $x_{ij} \geq 0$: Quantity supplied from warehouse $i$ to customer $j$.

##### Objective Function

Minimize total cost (fixed + transportation):

$$
\min \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in C} t_{ij} x_{ij}
$$

##### Constraints

1. **Demand Satisfaction:** Each customer's demand must be fully met.

   $$
   \sum_{i \in F} x_{ij} = d_j \quad \forall j \in C
   $$

   That is:
   - $x_{\text{S1},j} + x_{\text{S2},j} + x_{\text{S3},j} = d_j$ for $j = \text{C1}, \text{C2}, \text{C3}$

2. **Warehouse Activation:** No goods can be shipped from a warehouse unless it is open.

   $$
   x_{ij} \leq M_{ij} y_i \quad \forall i \in F, \forall j \in C
   $$

   Where $M_{ij}$ is a sufficiently large constant, e.g., $M_{ij} = d_j$ (since no customer can receive more than their demand from any warehouse):

   $$
   x_{ij} \leq d_j y_i \quad \forall i \in F, \forall j \in C
   $$

3. **Variable Domains:**

   $$
   y_i \in \{0,1\} \quad \forall i \in F
   $$
   $$
   x_{ij} \geq 0 \quad \forall i \in F, \forall j \in C
   $$

##### Complete Model

$$
\begin{align*}
\min \quad & 102.33\, y_{\text{S1}} + 94.92\, y_{\text{S2}} + 91.83\, y_{\text{S3}} \\
&+ 1506.22\, x_{\text{S1},\text{C1}} + 70.90\, x_{\text{S1},\text{C2}} + 8.44\, x_{\text{S1},\text{C3}} \\
&+ 1732.65\, x_{\text{S2},\text{C1}} + 1780.72\, x_{\text{S2},\text{C2}} + 567.44\, x_{\text{S2},\text{C3}} \\
&+ 115.66\, x_{\text{S3},\text{C1}} + 100.76\, x_{\text{S3},\text{C2}} + 64.68\, x_{\text{S3},\text{C3}} \\
\text{s.t.} \quad
& x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} + x_{\text{S3},\text{C1}} = 1083 \\
& x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} + x_{\text{S3},\text{C2}} = 776 \\
& x_{\text{S1},\text{C3}} + x_{\text{S2},\text{C3}} + x_{\text{S3},\text{C3}} = 16214 \\
& x_{\text{S1},j} \leq d_j\, y_{\text{S1}} \quad \forall j \in \{\text{C1},\text{C2},\text{C3}\} \\
& x_{\text{S2},j} \leq d_j\, y_{\text{S2}} \quad \forall j \in \{\text{C1},\text{C2},\text{C3}\} \\
& x_{\text{S3},j} \leq d_j\, y_{\text{S3}} \quad \forall j \in \{\text{C1},\text{C2},\text{C3}\} \\
& y_{\text{S1}}, y_{\text{S2}}, y_{\text{S3}} \in \{0,1\} \\
& x_{ij} \geq 0 \quad \forall i \in \{\text{S1},\text{S2},\text{S3}\},\, j \in \{\text{C1},\text{C2},\text{C3}\}
\end{align*}
$$