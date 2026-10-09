##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is activated.

##### Objective Function

$\min \sum_{i\in I}\sum_{j\in J} c_{ij}x_{ij} + \sum_{i\in I} f_i y_i$

##### Constraints

1. Supermarket demand: $\sum_{i\in I} x_{ij} = d_j,\quad \forall j \in J$
2. Domains: $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

##### Data Mapping

- $I = \{\text{S1}, \text{S2}\}$ (suppliers, from fixed_cost.csv and transportation_costs.csv, column "Unnamed: 0")
- $J = \{\text{C1}, \text{C2}\}$ (supermarkets, from demand.csv and transportation_costs.csv, columns "C1", "C2")
- $d_j$: demand for supermarket $j$ (from demand.csv, column "demand")
  - $d_{\text{C1}} = 144$
  - $d_{\text{C2}} = 216$
- $f_i$: fixed cost for supplier $i$ (from fixed_cost.csv, column "fixed_costs")
  - $f_{\text{S1}} = 105.97$
  - $f_{\text{S2}} = 85.31$
- $c_{ij}$: per-unit transportation cost from supplier $i$ to supermarket $j$ (from transportation_costs.csv)
  - $c_{\text{S1},\text{C1}} = 2358.39$
  - $c_{\text{S1},\text{C2}} = 1492.08$
  - $c_{\text{S2},\text{C1}} = 0.07$
  - $c_{\text{S2},\text{C2}} = 52.32$

##### Source-Column Data Mapping

- demand.csv: "customer" $\rightarrow$ $J$, "demand" $\rightarrow$ $d_j$
- fixed_cost.csv: "Unnamed: 0" $\rightarrow$ $I$, "fixed_costs" $\rightarrow$ $f_i$
- transportation_costs.csv: "Unnamed: 0" $\rightarrow$ $I$, "C1", "C2" $\rightarrow$ $c_{ij}$

All parameters are directly mapped from the provided CSV columns. No additional constraints or conditional bounds are imposed beyond those specified above.