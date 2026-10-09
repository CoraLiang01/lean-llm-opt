##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to branch $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is activated.

##### Objective Function

\[
\min \sum_{i\in I}\sum_{j\in J} c_{ij} x_{ij} + \sum_{i\in I} f_i y_i
\]

##### Constraints

1. **Branch demand:**  
   $\sum_{i\in I} x_{ij} = d_j,\quad \forall j\in J$

2. **Supplier activation (unconditional bounds):**  
   $x_{ij} \geq 0,\quad \forall i\in I,\, j\in J$  
   $y_i \in \{0,1\},\quad \forall i\in I$

##### Data Mapping

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$ (suppliers, from fixed_cost.csv and transportation_costs.csv)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}\}$ (branches/customers, from demand.csv and transportation_costs.csv)
- $d_j$ (demand for branch $j$), from demand.csv:
  - $d_{\text{C1}} = 143$
  - $d_{\text{C2}} = 6$
  - $d_{\text{C3}} = 10$
  - $d_{\text{C4}} = 25$
  - $d_{\text{C5}} = 3$
- $f_i$ (fixed cost for supplier $i$), from fixed_cost.csv:
  - $f_{\text{S1}} = 97.65$
  - $f_{\text{S2}} = 99.76$
  - $f_{\text{S3}} = 100.76$
  - $f_{\text{S4}} = 105.32$
  - $f_{\text{S5}} = 98.88$
- $c_{ij}$ (transportation cost per unit from supplier $i$ to branch $j$), from transportation_costs.csv:
  - $c_{\text{S1},\text{C1}} = 150.74$, $c_{\text{S1},\text{C2}} = 0.02$, $c_{\text{S1},\text{C3}} = 49.13$, $c_{\text{S1},\text{C4}} = 2080.15$, $c_{\text{S1},\text{C5}} = 426.4$
  - $c_{\text{S2},\text{C1}} = 233.05$, $c_{\text{S2},\text{C2}} = 97.73$, $c_{\text{S2},\text{C3}} = 49.84$, $c_{\text{S2},\text{C4}} = 1982.39$, $c_{\text{S2},\text{C5}} = 23.96$
  - $c_{\text{S3},\text{C1}} = 55.68$, $c_{\text{S3},\text{C2}} = 935.61$, $c_{\text{S3},\text{C3}} = 4.03$, $c_{\text{S3},\text{C4}} = 73.09$, $c_{\text{S3},\text{C5}} = 525.32$
  - $c_{\text{S4},\text{C1}} = 1483.82$, $c_{\text{S4},\text{C2}} = 1801.08$, $c_{\text{S4},\text{C3}} = 112.16$, $c_{\text{S4},\text{C4}} = 816.05$, $c_{\text{S4},\text{C5}} = 107.01$
  - $c_{\text{S5},\text{C1}} = 1119.47$, $c_{\text{S5},\text{C2}} = 884.31$, $c_{\text{S5},\text{C3}} = 0.08$, $c_{\text{S5},\text{C4}} = 1544.95$, $c_{\text{S5},\text{C5}} = 543.67$

##### Source-Column Data Mapping

- demand.csv: customer $\to$ $j$, demand $\to$ $d_j$
- fixed_cost.csv: Unnamed: 0 $\to$ $i$, fixed_costs $\to$ $f_i$
- transportation_costs.csv: Unnamed: 0 $\to$ $i$, $Ck$ columns $\to$ $c_{ij}$ for $j = Ck$