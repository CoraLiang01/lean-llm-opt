### Abstract Mathematical Model

#### Index Sets
- $F$: Set of suppliers (indexed by $i$)
- $C$: Set of stores/customers (indexed by $j$)

#### Parameters
- $f_i$: Fixed cost to activate supplier $i$  
  (Data: fixed_cost.csv, column "fixed_costs", table_id file_1_view_0)
- $t_{ij}$: Per-unit transportation cost from supplier $i$ to store $j$  
  (Data: transportation_costs.csv, columns "Unnamed: 0" (supplier), $C_j$ (store), table_id file_2_view_0)
- $d_j$: Demand at store $j$  
  (Data: demand.csv, column "demand", table_id file_0_view_0)

#### Decision Variables
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated, 0 otherwise
- $x_{ij} \geq 0$: Quantity supplied from supplier $i$ to store $j$

#### Objective
$$
\min \left( \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in C} t_{ij} x_{ij} \right)
$$

#### Constraints

1. **Demand Satisfaction:**  
  Each store's demand must be fully met:
$$
\sum_{i \in F} x_{ij} = d_j \quad \forall j \in C
$$

2. **Supplier Activation:**  
  A supplier can only ship if activated:
$$
\sum_{j \in C} x_{ij} \leq M_i y_i \quad \forall i \in F
$$
  where $M_i$ is a sufficiently large constant (e.g., $M_i = \sum_{j \in C} d_j$).

3. **Variable Domains:**  
$$
y_i \in \{0,1\} \quad \forall i \in F
$$
$$
x_{ij} \geq 0 \quad \forall i \in F,\, j \in C
$$

---

### Data Mapping

- **fixed_cost.csv** (table_id: file_1_view_0):  
  Supplier index: "Unnamed: 0"  
  Fixed cost: "fixed_costs"

- **transportation_costs.csv** (table_id: file_2_view_0):  
  Supplier index: "Unnamed: 0"  
  Store indices: "C1", "C2", ..., "C6" (column names)

- **demand.csv** (table_id: file_0_view_0):  
  Store index: "customer"  
  Demand: "demand"

All records from each file are included as returned by CSVQA. No additional filtering or aggregation is applied.