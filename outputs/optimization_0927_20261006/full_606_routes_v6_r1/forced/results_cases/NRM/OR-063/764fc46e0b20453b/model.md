#### Abstract Mathematical Model

**Index Sets:**
- $F$: set of warehouses (indexed by $i$), from `fixed_cost.csv` column `Unnamed: 0` and `transportation_costs.csv` row `Unnamed: 0`
- $C$: set of musicians/bands (indexed by $j$), from `demand.csv` column `customer` and `transportation_costs.csv` columns

**Parameters:**
- $f_i$: fixed cost to operate warehouse $i \in F$ (`fixed_cost.csv`, column `fixed_costs`)
- $c_{ij}$: transportation cost per unit from warehouse $i \in F$ to customer $j \in C$ (`transportation_costs.csv`, entry at row $i$, column $j$)
- $d_j$: demand of customer $j \in C$ (`demand.csv`, column `demand`)

**Decision Variables:**
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is operational, 0 otherwise
- $x_{ij} \geq 0$: quantity supplied from warehouse $i$ to customer $j$

**Objective:**
\[
\min \quad \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in C} c_{ij} x_{ij}
\]

**Constraints:**
1. **Demand Satisfaction:**  
   For all $j \in C$,
   \[
   \sum_{i \in F} x_{ij} = d_j
   \]
2. **Warehouse Activation:**  
   For all $i \in F$, $j \in C$,
   \[
   x_{ij} \leq d_j y_i
   \]
   (No supply from warehouse $i$ to customer $j$ unless warehouse $i$ is open.)

3. **Variable Domains:**  
   \[
   y_i \in \{0,1\} \quad \forall i \in F
   \]
   \[
   x_{ij} \geq 0 \quad \forall i \in F,\, j \in C
   \]

---

#### Data Mapping

- **Warehouses ($F$):**  
  - Source: `fixed_cost.csv`, column `Unnamed: 0`  
  - Source: `transportation_costs.csv`, row `Unnamed: 0`
- **Musicians/Bands ($C$):**  
  - Source: `demand.csv`, column `customer`  
  - Source: `transportation_costs.csv`, columns (excluding `Unnamed: 0`)
- **Fixed Costs ($f_i$):**  
  - Source: `fixed_cost.csv`, column `fixed_costs`
- **Transportation Costs ($c_{ij}$):**  
  - Source: `transportation_costs.csv`, entry at row `Unnamed: 0` = $i$, column $j$
- **Demand ($d_j$):**  
  - Source: `demand.csv`, column `demand`

No additional constraints or data sources are used beyond those specified in the query and returned by CSVQA.