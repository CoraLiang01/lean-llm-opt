#### Abstract Facility Location Model

**Index Sets**
- $I$: Set of potential factory sites (from `facility_costs.csv`, column `Facility`)
- $J$: Set of distribution centers (from `demand_requirements.csv`, column `Destination`)

**Parameters**
- $f_i$: Fixed cost to open factory $i \in I$ (from `facility_costs.csv`, column `FixedCost`)
- $c_{ij}$: Variable shipping cost per unit from factory $i \in I$ to distribution center $j \in J$ (from `shipping_costs.csv`, row `Origin` and columns $J$)
- $d_j$: Demand at distribution center $j \in J$ (from `demand_requirements.csv`, column `Demand`)
- $u_i$: Capacity of factory $i \in I$ (from `facility_costs.csv`, column `Capacity`)

**Decision Variables**
- $y_i \in \{0,1\}$: 1 if factory $i$ is constructed, 0 otherwise
- $x_{ij} \geq 0$: Quantity shipped from factory $i$ to distribution center $j$

**Objective**
\[
\min \quad \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

**Constraints**
1. **Demand Satisfaction:**  
   $\displaystyle \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J$

2. **Factory Capacity (if applicable):**  
   $\displaystyle \sum_{j \in J} x_{ij} \leq u_i y_i \quad \forall i \in I$

3. **Non-negativity and Binary:**  
   $x_{ij} \geq 0 \quad \forall i \in I,\, j \in J$  
   $y_i \in \{0,1\} \quad \forall i \in I$

---

#### Data Mapping

- **facility_costs.csv**
  - Table ID: `file_0_view_0`
  - Columns: `Facility` (factory index $I$), `FixedCost` (parameter $f_i$), `Capacity` (parameter $u_i$)
- **shipping_costs.csv**
  - Table ID: `file_1_view_0`
  - Row: `Origin` (factory index $I$)
  - Columns: $J$ (distribution center index), values: `c_{ij}` (shipping cost from $i$ to $j$)
- **demand_requirements.csv**
  - Table ID: `file_2_view_0`
  - Columns: `Destination` (distribution center index $J$), `Demand` (parameter $d_j$)

All records from each file are included as returned by CSVQA. No additional filtering or aggregation is applied.