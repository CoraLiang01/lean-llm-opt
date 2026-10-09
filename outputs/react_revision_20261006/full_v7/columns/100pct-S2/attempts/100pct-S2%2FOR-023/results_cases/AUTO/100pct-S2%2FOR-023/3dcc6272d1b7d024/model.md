##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Store demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Supplier activation logic:**  
   \[
   x_{ij} \leq D_j y_i, \quad \forall i \in I, \forall j \in J
   \]
   where $D_j$ is the demand of store $j$ (from data), ensuring $x_{ij}=0$ if $y_i=0$.
3. **Variable domains:**  
   \[
   x_{ij} \geq 0, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: Set of suppliers (facility names from `fixed_cost.csv` and rows of `transportation_costs.csv`).
- $J$: Set of stores (customer names from `demand.csv` and columns of `transportation_costs.csv`).
- $d_j$: Demand of store $j$ (from `demand.csv`, column "demand", indexed by "Customer").
- $f_i$: Fixed cost for supplier $i$ (from `fixed_cost.csv`, column "fixed_costs", indexed by "Unnamed: 2").
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$ (from `transportation_costs.csv`, row "Unnamed: 2" for supplier, column for store).

##### Data Mapping

- $I$: All unique values in `fixed_cost.csv`, column "Unnamed: 2" (`file_1_view_0`).
- $J$: All unique values in `demand.csv`, column "Customer" (`file_0_view_0`).
- $d_j$: `demand.csv`, column "demand", indexed by "Customer" (`file_0_view_0`).
- $f_i$: `fixed_cost.csv`, column "fixed_costs", indexed by "Unnamed: 2" (`file_1_view_0`).
- $c_{ij}$: `transportation_costs.csv`, row "Unnamed: 2" for supplier $i$, column for store $j$ (`file_2_view_0`).
- $x_{ij}$, $y_i$: Decision variables as defined above.

**Note:** The mapping between store names in `demand.csv` and columns in `transportation_costs.csv` must be aligned by the actual store names. If necessary, use the exact column names from `transportation_costs.csv` for $J$.

---

This model ensures that all store demands are met, suppliers are only activated if they ship goods, and the total cost (fixed plus transportation) is minimized. All parameters are directly mapped to the provided CSV data.