##### Decision Variables

$x_{ij} \geq 0$: quantity supplied from supplier $i \in I$ to supermarket $j \in J$ (continuous).  
$y_i \in \{0,1\}$: 1 if supplier $i$ is activated, 0 otherwise.

##### Parameters

Suppliers $I = \{\text{S1}, \text{S2}\}$  
Supermarkets $J = \{\text{C1}, \text{C2}\}$

Demands:
- $d_{\text{C1}} = 144$
- $d_{\text{C2}} = 216$

Fixed costs:
- $f_{\text{S1}} = 105.97$
- $f_{\text{S2}} = 85.31$

Transportation costs:
- $c_{\text{S1},\text{C1}} = 2358.39$
- $c_{\text{S1},\text{C2}} = 1492.08$
- $c_{\text{S2},\text{C1}} = 0.07$
- $c_{\text{S2},\text{C2}} = 52.32$

Let $M = d_{\text{C1}} + d_{\text{C2}} = 360$ (a valid upper bound for total supply from any supplier, since there are no explicit supplier capacity limits).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. Demand satisfaction for each supermarket:
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   That is,
   - $x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} = 144$
   - $x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} = 216$

2. Supplier activation: a supplier can only ship if activated:
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   That is,
   - $x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} \leq 360 y_{\text{S1}}$
   - $x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} \leq 360 y_{\text{S2}}$

3. Variable domains:
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Full Parameter Listing

- $I = \{\text{S1}, \text{S2}\}$
- $J = \{\text{C1}, \text{C2}\}$
- $d_{\text{C1}} = 144$, $d_{\text{C2}} = 216$
- $f_{\text{S1}} = 105.97$, $f_{\text{S2}} = 85.31$
- $c_{\text{S1},\text{C1}} = 2358.39$, $c_{\text{S1},\text{C2}} = 1492.08$
- $c_{\text{S2},\text{C1}} = 0.07$, $c_{\text{S2},\text{C2}} = 52.32$
- $M = 360$