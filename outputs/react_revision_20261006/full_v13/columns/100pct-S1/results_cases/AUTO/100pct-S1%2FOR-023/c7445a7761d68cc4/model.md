##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise (binary).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   (Each store's demand must be fully met.)

2. **Supplier activation logic:**  
   \[
   \sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I
   \]
   (A supplier can only ship if activated. $M_i$ is a valid upper bound, e.g., $M_i = \sum_{j \in J} d_j$.)

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: Set of suppliers (from `file_1_view_0`, column `Unnamed: 2`)
- $J$: Set of stores (from `file_0_view_0`, column `Customer`)
- $d_j$: Demand of store $j$ (from `file_0_view_0`, column `demand`)
- $f_i$: Fixed cost for supplier $i$ (from `file_1_view_0`, column `fixed_costs`)
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$ (from `file_2_view_0`, row `Unnamed: 2` = $i$, column = $j$)
- $M_i$: Big-M upper bound for supplier $i$ (e.g., $M_i = \sum_{j \in J} d_j$)

##### Data Mapping

- Suppliers $I$: `file_1_view_0`, column `Unnamed: 2`
- Stores $J$: `file_0_view_0`, column `Customer`
- Demand $d_j$: `file_0_view_0`, column `demand`
- Fixed cost $f_i$: `file_1_view_0`, column `fixed_costs`
- Transportation cost $c_{ij}$: `file_2_view_0`, rows indexed by `Unnamed: 2` (supplier), columns by store name (see below)
- $M_i$: $M_i = \sum_{j \in J} d_j$ (sum over all store demands from `file_0_view_0`, column `demand`)

**Note:**  
The mapping between store names in `file_0_view_0` (`Customer_1`, ..., `Customer_5`) and the columns in `file_2_view_0` (`BANCROFT`, `CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO`) must be established according to the actual store identities. If the mapping is direct, then $J = \{$BANCROFT, CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO$\}$ and $c_{ij}$ is taken directly from `file_2_view_0`. Otherwise, align store names accordingly.

---

This model determines which suppliers to activate and how much each should ship to each store to minimize total cost, subject to all demand being met and only activated suppliers shipping product. All parameters are mapped directly to the provided CSV data.