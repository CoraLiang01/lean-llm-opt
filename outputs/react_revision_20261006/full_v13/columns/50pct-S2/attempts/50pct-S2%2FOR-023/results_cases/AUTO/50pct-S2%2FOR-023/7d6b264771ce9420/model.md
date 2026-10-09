##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise (binary).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

where:
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$.
- $f_i$: Fixed cost to activate supplier $i$.

##### Constraints

1. **Demand satisfaction at each store:**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   where $d_j$ is the demand at store $j$.

2. **Supplier activation logic:**
   \[
   \sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I
   \]
   where $M_i$ is a sufficiently large upper bound for supplier $i$ (e.g., $M_i = \sum_{j \in J} d_j$).

3. **Variable domains:**
   \[
   x_{ij} \geq 0, \quad y_i \in \{0,1\}, \quad \forall i \in I, \forall j \in J
   \]

##### Index Sets and Data Mapping

- $I$: Set of suppliers (facilities), from `file_1_view_0` and `file_2_view_0` column "Unnamed: 1" and row "Unnamed: 0".
- $J$: Set of stores (customers), from `file_0_view_0` column "Customer" and `file_2_view_0` columns ["CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO", "BANCROFT"].
- $d_j$: Demand for store $j$, from `file_0_view_0`, column "demand", indexed by "Customer".
- $f_i$: Fixed cost for supplier $i$, from `file_1_view_0`, column "fixed_costs", indexed by "Unnamed: 1".
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$, from `file_2_view_0`, row "Unnamed: 0" (supplier), columns ["CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO", "BANCROFT"] (store).
- $M_i$: Big-M for each supplier $i$, set as $M_i = \sum_{j \in J} d_j$.

##### Data Mapping

- Suppliers $I$: `file_1_view_0`["Unnamed: 1"] and `file_2_view_0`["Unnamed: 0"]
- Stores $J$: `file_0_view_0`["Customer"] and `file_2_view_0`[columns: "CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO", "BANCROFT"]
- Demand $d_j$: `file_0_view_0`["demand"], indexed by "Customer"
- Fixed cost $f_i$: `file_1_view_0`["fixed_costs"], indexed by "Unnamed: 1"
- Transportation cost $c_{ij}$: `file_2_view_0`, rows "Unnamed: 0" (supplier), columns ["CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO", "BANCROFT"] (store)
- $M_i$: $M_i = \sum_{j \in J} d_j$ (computed from demand data)

All index sets and parameters are defined directly from the current CSV data.