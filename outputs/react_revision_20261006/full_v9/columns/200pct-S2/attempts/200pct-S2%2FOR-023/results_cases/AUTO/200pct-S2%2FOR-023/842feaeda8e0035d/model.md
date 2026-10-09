##### Decision Variables

$x_{ij} \geq 0$: quantity of goods shipped from supplier $i \in I$ to store $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is operational (open).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Store demand: $\sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J$
2. Supplier activation: $\sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I$
3. Domains: $x_{ij} \geq 0$ (continuous); $y_i \in \{0,1\}$

Where:
- $I$ is the set of suppliers (facility locations from fixed_cost.csv and transportation_costs.csv row labels)
- $J$ is the set of stores (store/customer names from demand.csv and transportation_costs.csv column labels)
- $c_{ij}$ is the per-unit transportation cost from supplier $i$ to store $j$ (from transportation_costs.csv)
- $f_i$ is the fixed cost for opening supplier $i$ (from fixed_cost.csv)
- $d_j$ is the demand at store $j$ (from demand.csv)
- $M_i$ is a sufficiently large upper bound for supplier $i$ (e.g., $M_i = \sum_{j \in J} d_j$ for all $i$ if no explicit supplier capacity is given)

##### Data Mapping

- $I$: All unique facility names from column "Unnamed: 3" in file_1_view_0 (fixed_cost.csv) and "Unnamed: 4" in file_2_view_0 (transportation_costs.csv)  
- $J$: All unique customer/store names from column "Customer" in file_0_view_0 (demand.csv) and columns ["BANCROFT", "CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO"] in file_2_view_0 (transportation_costs.csv)  
- $d_j$: Column "demand" in file_0_view_0, indexed by "Customer"  
- $f_i$: Column "fixed_costs" in file_1_view_0, indexed by "Unnamed: 3"  
- $c_{ij}$: Entry in file_2_view_0, row indexed by "Unnamed: 4" (supplier), column indexed by store name  
- $M_i$: $M_i = \sum_{j \in J} d_j$ for all $i$ (no explicit supplier capacity given)

All index sets and parameters are to be taken directly from the referenced columns and rows in the current CSV files.