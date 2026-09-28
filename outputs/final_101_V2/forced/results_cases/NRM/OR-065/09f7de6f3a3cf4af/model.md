#### Abstract Mathematical Model

**Index Sets:**
- $F$: set of warehouses (indexed by $i$), from fixed_cost.csv, column "Unnamed: 0"
- $S$: set of musicians/bands (indexed by $j$), from demand.csv, column "customer"

**Parameters:**
- $f_i$: fixed cost to open warehouse $i \in F$ (fixed_cost.csv, column "fixed_costs")
- $d_j$: demand of musician/band $j \in S$ (demand.csv, column "demand")
- $c_{ij}$: transportation cost per unit from warehouse $i \in F$ to musician/band $j \in S$ (transportation_costs.csv, columns "Unnamed: 0" for $i$, "C1", "C2", ... for $j$)

**Decision Variables:**
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise
- $x_{ij} \geq 0$: quantity supplied from warehouse $i$ to musician/band $j$

**Objective:**
\[
\min \quad \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in S} c_{ij} x_{ij}
\]

**Constraints:**
1. **Demand Satisfaction:**  
   $\sum_{i \in F} x_{ij} = d_j \quad \forall j \in S$

2. **Warehouse Activation:**  
   $x_{ij} \leq d_j y_i \quad \forall i \in F, \forall j \in S$

3. **Variable Domains:**  
   $y_i \in \{0,1\} \quad \forall i \in F$  
   $x_{ij} \geq 0 \quad \forall i \in F, \forall j \in S$

---

#### Data Mapping

- **Warehouses ($F$) and fixed costs ($f_i$):**  
  Source: fixed_cost.csv, table_id file_1_view_0, columns "Unnamed: 0" (warehouse id), "fixed_costs"
- **Musicians/Bands ($S$) and demands ($d_j$):**  
  Source: demand.csv, table_id file_0_view_0, columns "customer" (musician/band id), "demand"
- **Transportation costs ($c_{ij}$):**  
  Source: transportation_costs.csv, table_id file_2_view_0, rows indexed by "Unnamed: 0" (warehouse id), columns "C1", "C2", "C3", ... (musician/band id)

---

This model determines which warehouses to open and how much each should supply to each musician/band to minimize total fixed and transportation costs, subject to demand fulfillment and warehouse activation logic.