#### Abstract Mathematical Model

**Index Sets:**
- $F$: set of suppliers (indexed by $i$), from `fixed_cost.csv` column `Unnamed: 0` and `transportation_costs.csv` row `Unnamed: 0`
- $S$: set of supermarkets (indexed by $j$), from `demand.csv` column `customer` and `transportation_costs.csv` columns

**Parameters:**
- $f_i$: fixed cost of opening supplier $i \in F$ (`fixed_cost.csv`, column `fixed_costs`)
- $c_{ij}$: transportation cost per unit from supplier $i \in F$ to supermarket $j \in S$ (`transportation_costs.csv`, entry at row $i$, column $j$)
- $d_j$: demand of supermarket $j \in S$ (`demand.csv`, column `demand`)

**Decision Variables:**
- $y_i \in \{0,1\}$: 1 if supplier $i$ is open, 0 otherwise
- $x_{ij} \geq 0$: quantity supplied from supplier $i$ to supermarket $j$

**Objective:**
\[
\min \left( \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in S} c_{ij} x_{ij} \right)
\]

**Constraints:**
1. **Demand Satisfaction:**  
   For all $j \in S$,
   \[
   \sum_{i \in F} x_{ij} = d_j
   \]
2. **Supplier Activation:**  
   For all $i \in F$, $j \in S$,
   \[
   x_{ij} \leq d_j y_i
   \]
3. **Variable Domains:**  
   \[
   y_i \in \{0,1\} \quad \forall i \in F
   \]
   \[
   x_{ij} \geq 0 \quad \forall i \in F,\, j \in S
   \]

---

#### Data Mapping

- **Supplier set $F$:**  
  - `fixed_cost.csv`, column `Unnamed: 0`  
  - `transportation_costs.csv`, row `Unnamed: 0`
- **Supermarket set $S$:**  
  - `demand.csv`, column `customer`  
  - `transportation_costs.csv`, columns (excluding `Unnamed: 0`)
- **Fixed costs $f_i$:**  
  - `fixed_cost.csv`, column `fixed_costs`
- **Transportation costs $c_{ij}$:**  
  - `transportation_costs.csv`, entry at row `Unnamed: 0` = $i$, column $j$
- **Demands $d_j$:**  
  - `demand.csv`, column `demand`