#### Abstract Mathematical Model

**Index Sets:**
- $F$: Set of suppliers (indexed by $i$)
- $S$: Set of stores/customers (indexed by $j$)

**Parameters:**
- $f_i$: Fixed cost to open supplier $i$ (from `fixed_cost.csv`)
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$ (from `transportation_costs.csv`)
- $d_j$: Demand at store $j$ (from `demand.csv`)

**Decision Variables:**
- $y_i \in \{0,1\}$: 1 if supplier $i$ is open, 0 otherwise
- $x_{ij} \geq 0$: Quantity supplied from supplier $i$ to store $j$

**Objective:**
\[
\min \left( \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in S} c_{ij} x_{ij} \right)
\]

**Constraints:**
1. **Demand Satisfaction:**
   \[
   \sum_{i \in F} x_{ij} = d_j \quad \forall j \in S
   \]
2. **Supplier Activation:**
   \[
   x_{ij} \leq M_{ij} y_i \quad \forall i \in F, \forall j \in S
   \]
   where $M_{ij}$ is a sufficiently large constant (e.g., $M_{ij} \geq d_j$).

3. **Variable Domains:**
   \[
   y_i \in \{0,1\} \quad \forall i \in F
   \]
   \[
   x_{ij} \geq 0 \quad \forall i \in F, \forall j \in S
   \]

---

#### Data Mapping

- **Supplier fixed costs:**  
  Table: `fixed_cost.csv`  
  Columns: `Unnamed: 0` (supplier identifier), `fixed_costs` (parameter $f_i$)

- **Transportation costs:**  
  Table: `transportation_costs.csv`  
  Columns: `Unnamed: 0` (supplier identifier), one column per store (store identifiers; parameter $c_{ij}$)

- **Store demands:**  
  Table: `demand.csv`  
  Columns: `Customer` (store identifier), `demand` (parameter $d_j$)

---

**Note:**  
All index sets, parameters, and mappings are defined symbolically and correspond directly to the columns and tables listed above. No literal data values or record counts are included. The model is abstract and ready for instantiation with the provided data.