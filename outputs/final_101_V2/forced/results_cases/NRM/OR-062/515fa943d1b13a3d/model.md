### Abstract Mathematical Model

#### Index Sets
- $F$: set of suppliers (indexed by $i$)
- $S$: set of stores/customers (indexed by $j$)

#### Parameters
- $f_i$: fixed cost to open supplier $i$  
  (Data: table_id="file_1_view_0", column="fixed_costs", key column="Unnamed: 0")
- $c_{ij}$: per-unit transportation cost from supplier $i$ to store $j$  
  (Data: table_id="file_2_view_0", row key="Unnamed: 0" for supplier, column key for store)
- $d_j$: demand at store $j$  
  (Data: table_id="file_0_view_0", column="demand", key column="Customer")

#### Decision Variables
- $y_i \in \{0,1\}$: 1 if supplier $i$ is open, 0 otherwise
- $x_{ij} \geq 0$: quantity shipped from supplier $i$ to store $j$

#### Objective
$$
\min \left( \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in S} c_{ij} x_{ij} \right)
$$

#### Constraints

1. **Demand Satisfaction:**  
  $\sum_{i \in F} x_{ij} = d_j \quad \forall j \in S$

2. **Supplier Activation:**  
  $x_{ij} \leq M_{ij} y_i \quad \forall i \in F, \forall j \in S$  
  (where $M_{ij}$ is any sufficiently large constant, e.g., $M_{ij} \geq d_j$)

3. **Variable Domains:**  
  $y_i \in \{0,1\} \quad \forall i \in F$  
  $x_{ij} \geq 0 \quad \forall i \in F, \forall j \in S$

---

### Data Mapping

- **Suppliers ($F$):**  
  Identifiers from table_id="file_1_view_0", column="Unnamed: 0" (fixed_cost.csv)  
  Also used as row keys in table_id="file_2_view_0" (transportation_costs.csv)

- **Stores ($S$):**  
  Identifiers from table_id="file_0_view_0", column="Customer" (demand.csv)  
  Also used as column keys in table_id="file_2_view_0" (transportation_costs.csv)

- **Fixed Costs ($f_i$):**  
  table_id="file_1_view_0", column="fixed_costs", key column="Unnamed: 0"

- **Transportation Costs ($c_{ij}$):**  
  table_id="file_2_view_0", row key="Unnamed: 0" (supplier), column key (store)

- **Demands ($d_j$):**  
  table_id="file_0_view_0", column="demand", key column="Customer"

---

This model ensures all store demands are met, suppliers are only activated if used, and total cost (fixed plus transportation) is minimized.