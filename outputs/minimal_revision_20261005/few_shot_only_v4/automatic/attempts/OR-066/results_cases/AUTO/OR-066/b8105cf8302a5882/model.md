##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is activated.

##### Objective Function

\[
\min \sum_{i\in I}\sum_{j\in J} c_{ij} x_{ij} + \sum_{i\in I} f_i y_i
\]

##### Constraints

1. Supermarket demand satisfaction:  
   \[
   \sum_{i\in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. Supplier activation:  
   \[
   \sum_{j\in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j\in J} d_j = 144 + 216 = 360$.
3. Variable domains:  
   \[
   x_{ij} \geq 0 \text{ (continuous)}, \quad y_i \in \{0,1\}
   \]

##### Parameters

- $I = \{\text{S1}, \text{S2}\}$ (suppliers)
- $J = \{\text{C1}, \text{C2}\}$ (supermarkets)
- Demands:  
  $d_{\text{C1}} = 144$  
  $d_{\text{C2}} = 216$
- Fixed costs:  
  $f_{\text{S1}} = 105.97$  
  $f_{\text{S2}} = 85.31$
- Transportation costs:  
  $c_{\text{S1},\text{C1}} = 2358.39$  
  $c_{\text{S1},\text{C2}} = 1492.08$  
  $c_{\text{S2},\text{C1}} = 0.07$  
  $c_{\text{S2},\text{C2}} = 52.32$

##### Data Mapping

- demand.csv:  
  Columns: customer, demand  
  Mapping: $d_j$ for $j \in J$
- fixed_cost.csv:  
  Columns: Unnamed: 0 (supplier id), fixed_costs  
  Mapping: $f_i$ for $i \in I$
- transportation_costs.csv:  
  Rows: Unnamed: 0 (supplier id), Columns: C1, C2  
  Mapping: $c_{ij}$ for $i \in I$, $j \in J$