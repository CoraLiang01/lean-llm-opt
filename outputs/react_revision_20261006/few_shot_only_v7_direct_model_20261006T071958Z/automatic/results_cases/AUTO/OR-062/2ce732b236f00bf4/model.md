##### Decision Variables

$x_{ij} \geq 0$: quantity of liquor product shipped from supplier $i \in I$ to store $j \in J$ (continuous).

$y_i \in \{0,1\}$: whether supplier $i$ is activated (binary).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Store demand satisfaction:
   $$
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   $$
2. Domains:
   $$
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   $$
   $$
   y_i \in \{0,1\}, \quad \forall i \in I
   $$

##### Index Sets and Parameter Mapping

- $I$: Set of suppliers, from column `Unnamed: 0` in table_id `file_1_view_0` (`fixed_cost.csv`)
- $J$: Set of stores, from column `Customer` in table_id `file_0_view_0` (`demand.csv`)
- $d_j$: Demand of store $j$, from column `demand` in table_id `file_0_view_0` (`demand.csv`)
- $f_i$: Fixed cost for supplier $i$, from column `fixed_costs` in table_id `file_1_view_0` (`fixed_cost.csv`)
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$, mapped as follows:
    - Supplier $i$ from column `Unnamed: 0` in table_id `file_2_view_0` (`transportation_costs.csv`)
    - Store $j$ from columns `CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO`, `BANCROFT` in table_id `file_2_view_0` (`transportation_costs.csv`)

##### Data Mapping

- Suppliers $I$: `file_1_view_0`, column `Unnamed: 0`
- Stores $J$: `file_0_view_0`, column `Customer`
- Demand $d_j$: `file_0_view_0`, columns `Customer`, `demand`
- Fixed cost $f_i$: `file_1_view_0`, columns `Unnamed: 0`, `fixed_costs`
- Transportation cost $c_{ij}$: `file_2_view_0`, row `Unnamed: 0` (supplier), columns `CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO`, `BANCROFT` (stores)

All parameters and sets are defined directly from the supplied CSV data. No additional constraints or bounds are imposed beyond those specified above.