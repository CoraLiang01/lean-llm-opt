#### Index Sets

- $F$: set of suppliers (from "fixed_cost.csv" Unnamed: 0 and "transportation_costs.csv" Unnamed: 0)
- $C$: set of supermarkets/customers (from "demand.csv" customer and "transportation_costs.csv" columns C1, ..., C12)

#### Parameters

- $f_i$: fixed cost of opening supplier $i \in F$ (from "fixed_cost.csv" fixed_costs)
- $t_{ij}$: transportation cost per unit from supplier $i \in F$ to supermarket $j \in C$ (from "transportation_costs.csv" columns C1, ..., C12)
- $d_j$: demand of supermarket $j \in C$ (from "demand.csv" demand)

#### Decision Variables

- $y_i \in \{0,1\}$: 1 if supplier $i$ is open, 0 otherwise
- $x_{ij} \geq 0$: quantity supplied from supplier $i$ to supermarket $j$

#### Objective

Minimize total cost (fixed + transportation):
$$
\min \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in C} t_{ij} x_{ij}
$$

#### Constraints

1. **Demand Satisfaction** (each supermarket's demand must be met):
   $$
   \sum_{i \in F} x_{ij} = d_j \quad \forall j \in C
   $$

2. **Supplier Activation** (no supply from unopened suppliers):
   $$
   x_{ij} \leq d_j y_i \quad \forall i \in F, \forall j \in C
   $$

3. **Variable Domains**:
   $$
   y_i \in \{0,1\} \quad \forall i \in F
   $$
   $$
   x_{ij} \geq 0 \quad \forall i \in F, \forall j \in C
   $$

---

#### Data Mapping

- "fixed_cost.csv" (table_id: file_1_view_0): supplier index set $F$ from column Unnamed: 0, fixed cost $f_i$ from column fixed_costs.
- "demand.csv" (table_id: file_0_view_0): supermarket/customer index set $C$ from column customer, demand $d_j$ from column demand.
- "transportation_costs.csv" (table_id: file_2_view_0): supplier index set $F$ from column Unnamed: 0, supermarket/customer index set $C$ from columns C1, ..., C12, transportation cost $t_{ij}$ from corresponding cell.

Returned records: all rows and columns from each file, as provided by CSVQA. No additional filtering applied.