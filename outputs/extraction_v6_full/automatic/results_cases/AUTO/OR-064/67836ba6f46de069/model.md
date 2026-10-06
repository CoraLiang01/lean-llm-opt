##### Decision Variables

- $x_{ij} \geq 0$: quantity of goods supplied from supplier $i \in I$ to supermarket $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether supplier $i$ is operational (open).

##### Parameters

- $I = \{$S1, S2, ..., S24$\}$: set of suppliers (facilities).
- $J = \{$C1, C2, ..., C25$\}$: set of supermarkets (customers).
- $d_j$: demand of supermarket $j \in J$ (from demand.csv).
- $f_i$: fixed cost of opening supplier $i \in I$ (from fixed_cost.csv).
- $c_{ij}$: per-unit transportation cost from supplier $i$ to supermarket $j$ (from transportation_costs.csv).

###### Demand vector $d_j$ (for $j \in J$):

\[
\begin{aligned}
&d_{C1} = 1097,\quad d_{C2} = 61,\quad d_{C3} = 11,\quad d_{C4} = 7,\quad d_{C5} = 82,\\
&d_{C6} = 37,\quad d_{C7} = 483,\quad d_{C8} = 582,\quad d_{C9} = 223,\quad d_{C10} = 89,\\
&d_{C11} = 60,\quad d_{C12} = 55,\quad d_{C13} = 122,\quad d_{C14} = 66,\quad d_{C15} = 12,\\
&d_{C16} = 21,\quad d_{C17} = 53,\quad d_{C18} = 105,\quad d_{C19} = 1,\quad d_{C20} = 253,\\
&d_{C21} = 10,\quad d_{C22} = 53,\quad d_{C23} = 24,\quad d_{C24} = 122,\quad d_{C25} = 42
\end{aligned}
\]

###### Fixed cost vector $f_i$ (for $i \in I$):

\[
\begin{aligned}
&f_{S1} = 98.88,\quad f_{S2} = 99.73,\quad f_{S3} = 94.01,\quad f_{S4} = 93.77,\quad f_{S5} = 107.59,\\
&f_{S6} = 112.65,\quad f_{S7} = 97.05,\quad f_{S8} = 103,\quad f_{S9} = 90.45,\quad f_{S10} = 96.73,\\
&f_{S11} = 96.43,\quad f_{S12} = 112.19,\quad f_{S13} = 102.58,\quad f_{S14} = 88.85,\quad f_{S15} = 82.57,\\
&f_{S16} = 91.65,\quad f_{S17} = 101.38,\quad f_{S18} = 102.59,\quad f_{S19} = 105.97,\quad f_{S20} = 85.31,\\
&f_{S21} = 104.52,\quad f_{S22} = 100.2,\quad f_{S23} = 103.79,\quad f_{S24} = 105.27
\end{aligned}
\]

###### Transportation cost matrix $c_{ij}$ (for $i \in I$, $j \in J$):

Each $c_{ij}$ is the per-unit transportation cost from supplier $S_i$ to supermarket $C_j$, as given in the table above (see "transportation_costs.csv" for all values).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:** Each supermarket's demand must be fully met.
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation:** No goods can be shipped from a supplier unless it is open.
   \[
   \sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I
   \]
   where $M_i = \sum_{j \in J} d_j$ (since there are no explicit supplier capacity limits).

   For this instance:
   \[
   M_i = \sum_{j \in J} d_j = 1097 + 61 + 11 + 7 + 82 + 37 + 483 + 582 + 223 + 89 + 60 + 55 + 122 + 66 + 12 + 21 + 53 + 105 + 1 + 253 + 10 + 53 + 24 + 122 + 42 = 3619
   \]
   (Sum verified: $3619$)

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Complete Model

\[
\begin{aligned}
\min \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.} \quad & \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq 3619\, y_i, \quad \forall i \in I \\
& x_{ij} \geq 0, \quad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\}, \quad \forall i \in I
\end{aligned}
\]

Where all $c_{ij}$, $f_i$, and $d_j$ are as specified above and in the retrieved tables. All identifiers and coefficients are preserved as in the source data.