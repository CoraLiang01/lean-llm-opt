##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether warehouse $i$ is activated.

##### Objective Function

\[
\min \sum_{i\in I}\sum_{j\in J} c_{ij} x_{ij} + \sum_{i\in I} f_i y_i
\]

##### Constraints

1. Demand satisfaction:  
   \[
   \sum_{i\in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. Unconditional shipment bounds:  
   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]
3. Warehouse activation:  
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

##### Data Mapping

- $I = \{\text{S1}, \text{S2}, \text{S3}\}$ (warehouses, from fixed_cost.csv and transportation_costs.csv rows)
- $J = \{\text{C1}, \text{C2}, \text{C3}\}$ (musicians/bands, from demand.csv and transportation_costs.csv columns)
- $d_j$ (demand for each musician/band $j$), from demand.csv:
    - $d_{\text{C1}} = 1083$
    - $d_{\text{C2}} = 776$
    - $d_{\text{C3}} = 16214$
- $f_i$ (fixed cost for each warehouse $i$), from fixed_cost.csv:
    - $f_{\text{S1}} = 102.33$
    - $f_{\text{S2}} = 94.92$
    - $f_{\text{S3}} = 91.83$
- $c_{ij}$ (unit transportation cost from warehouse $i$ to musician/band $j$), from transportation_costs.csv:
    - $c_{\text{S1},\text{C1}} = 1506.22$, $c_{\text{S1},\text{C2}} = 70.90$, $c_{\text{S1},\text{C3}} = 8.44$
    - $c_{\text{S2},\text{C1}} = 1732.65$, $c_{\text{S2},\text{C2}} = 1780.72$, $c_{\text{S2},\text{C3}} = 567.44$
    - $c_{\text{S3},\text{C1}} = 115.66$, $c_{\text{S3},\text{C2}} = 100.76$, $c_{\text{S3},\text{C3}} = 64.68$

##### Source-column Data Mapping

- demand.csv: customer $\to$ $j$, demand $\to$ $d_j$
- fixed_cost.csv: Unnamed: 0 $\to$ $i$, fixed_costs $\to$ $f_i$
- transportation_costs.csv: Unnamed: 0 $\to$ $i$, C1/C2/C3 $\to$ $c_{ij}$

All parameters and indices are mapped directly from the provided CSV columns and rows.