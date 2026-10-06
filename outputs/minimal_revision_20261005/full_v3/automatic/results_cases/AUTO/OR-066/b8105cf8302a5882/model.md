##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from supplier $i$ to supermarket $j$ (continuous), for all $i$ in suppliers and $j$ in supermarkets.
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated, 0 otherwise.

##### Objective Function

\[
\min \sum_{i} \sum_{j} c_{ij} x_{ij} + \sum_{i} f_i y_i
\]

where $c_{ij}$ is the per-unit transportation cost from supplier $i$ to supermarket $j$, and $f_i$ is the fixed cost for activating supplier $i$.

##### Constraints

1. **Supermarket Demand Satisfaction**  
   \[
   \sum_{i} x_{ij} = d_j \quad \forall j
   \]
   Each supermarket $j$ must receive exactly its demand $d_j$.

2. **Supplier Activation Constraint**  
   \[
   \sum_{j} x_{ij} \leq M y_i \quad \forall i
   \]
   A supplier $i$ can only ship goods if it is activated ($y_i=1$). $M$ is a sufficiently large constant, here $M = \sum_j d_j$.

3. **Variable Domains**  
   \[
   x_{ij} \geq 0 \quad \forall i, j
   \]
   \[
   y_i \in \{0,1\} \quad \forall i
   \]

##### Data Mapping

- **Suppliers ($i$):**  
  Row identifiers from `file_1_view_0` ("Unnamed: 0"): S1, S2

- **Supermarkets ($j$):**  
  Column identifiers from `file_0_view_0` ("customer"): C1, C2

- **Demands ($d_j$):**  
  $d_j$ from `file_0_view_0`, column "demand", indexed by "customer".

- **Fixed Costs ($f_i$):**  
  $f_i$ from `file_1_view_0`, column "fixed_costs", indexed by "Unnamed: 0".

- **Transportation Costs ($c_{ij}$):**  
  $c_{ij}$ from `file_2_view_0`, rows indexed by "Unnamed: 0" (S1, S2), columns by "C1", "C2".

- **Big-M ($M$):**  
  $M = \sum_j d_j$, where $d_j$ as above.

##### Symbolic Model

\[
\begin{align*}
\min_{x_{ij}, y_i} \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.} \quad
& \sum_{i \in I} x_{ij} = d_j \quad && \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq M y_i \quad && \forall i \in I \\
& x_{ij} \geq 0 \quad && \forall i \in I,\, j \in J \\
& y_i \in \{0,1\} \quad && \forall i \in I
\end{align*}
\]

##### Data Mapping

- $I$: supplier set from `file_1_view_0` ("Unnamed: 0")
- $J$: supermarket set from `file_0_view_0` ("customer")
- $d_j$: `file_0_view_0`, column "demand", indexed by "customer"
- $f_i$: `file_1_view_0`, column "fixed_costs", indexed by "Unnamed: 0"
- $c_{ij}$: `file_2_view_0`, rows "Unnamed: 0", columns "C1", "C2"
- $M = \sum_j d_j$ (sum over all $d_j$ as above)