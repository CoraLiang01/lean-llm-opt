##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to customer $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether warehouse $i$ is opened.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   (Each customer's demand must be fully met.)

2. **Warehouse capacity:**  
   \[
   \sum_{j \in J} x_{ij} \leq \text{cap}_i y_i, \quad \forall i \in I
   \]
   (A warehouse cannot ship more than its capacity, and only if it is open.)

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: set of warehouses, from column `"Warehouse ID"` in table_id `"file_1_view_0"`.
- $J$: set of customers, from column `"Customer ID"` in table_id `"file_2_view_0"`.
- $c_{ij}$: unit transportation cost from warehouse $i$ to customer $j$, from table_id `"file_0_view_0"`, row `"Warehouse ID" = i"`, column $j$.
- $f_i$: fixed opening cost for warehouse $i$, from table_id `"file_1_view_0"`, row `"Warehouse ID" = i"`, column `"Fixed_Cost"`.
- $\text{cap}_i$: capacity of warehouse $i$, from table_id `"file_1_view_0"`, row `"Warehouse ID" = i"`, column `"Capacity"`.
- $d_j$: demand of customer $j$, from table_id `"file_2_view_0"`, row `"Customer ID" = j"`, column `"Demand"`.

##### Data Mapping

- $I$: `"Warehouse ID"` in `"file_1_view_0"` (`/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/warehouse.csv`)
- $J$: `"Customer ID"` in `"file_2_view_0"` (`/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/demand.csv`)
- $c_{ij}$: `"file_0_view_0"` (`/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/cost.csv`), row `"Warehouse ID" = i"`, column $j$
- $f_i$: `"file_1_view_0"`, row `"Warehouse ID" = i"`, column `"Fixed_Cost"`
- $\text{cap}_i$: `"file_1_view_0"`, row `"Warehouse ID" = i"`, column `"Capacity"`
- $d_j$: `"file_2_view_0"`, row `"Customer ID" = j"`, column `"Demand"`