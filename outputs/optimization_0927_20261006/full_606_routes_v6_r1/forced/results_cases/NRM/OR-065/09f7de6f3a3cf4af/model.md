#### Abstract Mathematical Model

**Index Sets:**
- $F$: set of warehouses (indexed by $i$)
- $C$: set of musicians or bands (indexed by $j$)

**Parameters:**
- $f_i$: fixed cost to operate warehouse $i$ (from `fixed_cost.csv`, column `fixed_costs`, table_id: `file_1_view_0`)
- $t_{ij}$: transportation cost per unit from warehouse $i$ to customer $j$ (from `transportation_costs.csv`, columns $C_j$, table_id: `file_2_view_0`)
- $d_j$: demand of customer $j$ (from `demand.csv`, column `demand`, table_id: `file_0_view_0`)

**Decision Variables:**
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is operational, 0 otherwise
- $x_{ij} \geq 0$: quantity supplied from warehouse $i$ to customer $j$

**Objective:**
\[
\min \quad \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in C} t_{ij} x_{ij}
\]

**Constraints:**
1. **Demand Satisfaction:**
   \[
   \sum_{i \in F} x_{ij} = d_j \quad \forall j \in C
   \]
2. **Warehouse Activation:**
   \[
   x_{ij} \leq d_j y_i \quad \forall i \in F, \forall j \in C
   \]
3. **Variable Domains:**
   \[
   y_i \in \{0,1\} \quad \forall i \in F
   \]
   \[
   x_{ij} \geq 0 \quad \forall i \in F, \forall j \in C
   \]

---

#### Data Mapping

- **Warehouses ($F$):** Identifiers from `fixed_cost.csv`, column `Unnamed: 0`, table_id: `file_1_view_0`
- **Customers ($C$):** Identifiers from `demand.csv`, column `customer`, table_id: `file_0_view_0`
- **Fixed Costs ($f_i$):** `fixed_cost.csv`, column `fixed_costs`, table_id: `file_1_view_0`
- **Demands ($d_j$):** `demand.csv`, column `demand`, table_id: `file_0_view_0`
- **Transportation Costs ($t_{ij}$):** `transportation_costs.csv`, columns $C_j$, table_id: `file_2_view_0`, with warehouse index from `Unnamed: 0` and customer index from column headers

All data is used as returned by the query, with no additional filtering or aggregation.