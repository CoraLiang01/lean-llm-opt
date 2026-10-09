##### Sets
- Suppliers: $I = \{S1, S2\}$
- Supermarkets: $J = \{C1, C2\}$

##### Parameters
- Demand for each supermarket:
  - $d_{C1} = 144$
  - $d_{C2} = 216$
- Fixed cost for each supplier:
  - $f_{S1} = 105.97$
  - $f_{S2} = 85.31$
- Per-unit transportation cost from supplier $i$ to supermarket $j$ ($c_{ij}$):

|        | C1      | C2      |
|--------|---------|---------|
| S1     | 2358.39 | 1492.08 |
| S2     | 0.07    | 52.32   |

##### Decision Variables
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated, 0 otherwise, for $i \in I$
- $x_{ij} \geq 0$: quantity supplied from supplier $i$ to supermarket $j$, for $i \in I$, $j \in J$

##### Objective Function
Minimize total cost:
$$
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$
That is,
$$
\min\ 105.97\,y_{S1} + 85.31\,y_{S2} + 2358.39\,x_{S1,C1} + 1492.08\,x_{S1,C2} + 0.07\,x_{S2,C1} + 52.32\,x_{S2,C2}
$$

##### Constraints

1. **Demand satisfaction:** Each supermarket must receive its full demand.
   $$
   \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J
   $$
   Specifically,
   $$
   x_{S1,C1} + x_{S2,C1} = 144
   $$
   $$
   x_{S1,C2} + x_{S2,C2} = 216
   $$

2. **Supplier activation:** A supplier can only supply if it is activated.
   $$
   x_{ij} \leq d_j y_i \quad \forall i \in I,\, j \in J
   $$
   That is,
   $$
   x_{S1,C1} \leq 144\,y_{S1}
   $$
   $$
   x_{S1,C2} \leq 216\,y_{S1}
   $$
   $$
   x_{S2,C1} \leq 144\,y_{S2}
   $$
   $$
   x_{S2,C2} \leq 216\,y_{S2}
   $$

3. **Variable domains:**
   $$
   y_i \in \{0,1\} \quad \forall i \in I
   $$
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

##### Complete Model

Minimize
$$
105.97\,y_{S1} + 85.31\,y_{S2} + 2358.39\,x_{S1,C1} + 1492.08\,x_{S1,C2} + 0.07\,x_{S2,C1} + 52.32\,x_{S2,C2}
$$

Subject to
\[
\begin{align*}
x_{S1,C1} + x_{S2,C1} &= 144 \\
x_{S1,C2} + x_{S2,C2} &= 216 \\
x_{S1,C1} &\leq 144\,y_{S1} \\
x_{S1,C2} &\leq 216\,y_{S1} \\
x_{S2,C1} &\leq 144\,y_{S2} \\
x_{S2,C2} &\leq 216\,y_{S2} \\
y_{S1},\,y_{S2} &\in \{0,1\} \\
x_{S1,C1},\,x_{S1,C2},\,x_{S2,C1},\,x_{S2,C2} &\geq 0
\end{align*}
\]