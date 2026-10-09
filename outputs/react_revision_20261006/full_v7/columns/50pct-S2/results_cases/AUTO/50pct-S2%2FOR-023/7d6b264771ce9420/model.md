##### Decision Variables

$x_{ij} \geq 0$: quantity of goods shipped from supplier $i \in I$ to store $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is operational (binary).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Store demand: $\sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J$
2. Supplier activation: $\sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I$
3. Domains: $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

Where:
- $I$ is the set of suppliers (facility locations from fixed_cost.csv and transportation_costs.csv row IDs)
- $J$ is the set of stores (store/customer IDs from demand.csv and transportation_costs.csv column IDs)
- $c_{ij}$ is the transportation cost per unit from supplier $i$ to store $j$ (from transportation_costs.csv)
- $f_i$ is the fixed cost for opening supplier $i$ (from fixed_cost.csv)
- $d_j$ is the demand at store $j$ (from demand.csv)
- $M_i$ is a sufficiently large upper bound for supplier $i$ (e.g., $M_i = \sum_{j \in J} d_j$ for all $i$ if no explicit supplier capacity is given)

##### Data Mapping

- $I$: All unique values in column "Unnamed: 1" of table_id file_1_view_0 (fixed_cost.csv), and row IDs in "Unnamed: 0" of table_id file_2_view_0 (transportation_costs.csv)
- $J$: All unique values in column "Customer" of table_id file_0_view_0 (demand.csv), and columns ["CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO", "BANCROFT"] in table_id file_2_view_0 (transportation_costs.csv)
- $d_j$: Column "demand" in table_id file_0_view_0, indexed by "Customer"
- $f_i$: Column "fixed_costs" in table_id file_1_view_0, indexed by "Unnamed: 1"
- $c_{ij}$: Entry in table_id file_2_view_0, row "Unnamed: 0" = $i$, column = $j$
- $M_i$: $M_i = \sum_{j \in J} d_j$ for all $i$ (since no explicit supplier capacity is given)

All index sets and parameters are to be taken directly from the referenced columns and rows in the source tables.