#### Index Sets

- $I$: set of suppliers (from file_1_view_0["Unnamed: 0"], e.g., S1, S2)
- $J$: set of supermarkets (from file_0_view_0["customer"], e.g., C1, C2)

#### Parameters

- $f_i$: fixed cost for activating supplier $i \in I$ (from file_1_view_0["fixed_costs"])
- $c_{ij}$: per-unit transportation cost from supplier $i \in I$ to supermarket $j \in J$ (from file_2_view_0, columns "C1", "C2", rows indexed by "Unnamed: 0")
- $d_j$: demand at supermarket $j \in J$ (from file_0_view_0["demand"])

#### Decision Variables

- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated, 0 otherwise
- $x_{ij} \geq 0$: amount supplied from supplier $i$ to supermarket $j$

#### Objective

$$
\min \left( \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \right)
$$

#### Constraints

1. **Demand Satisfaction:**  
   $$
   \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J
   $$

2. **Supplier Activation:**  
   $$
   x_{ij} \leq d_j y_i \quad \forall i \in I, \forall j \in J
   $$

3. **Variable Domains:**  
   $$
   y_i \in \{0,1\} \quad \forall i \in I
   $$
   $$
   x_{ij} \geq 0 \quad \forall i \in I, \forall j \in J
   $$

---

#### Data Mapping

- file_1_view_0["Unnamed: 0"]: supplier index $I$
- file_1_view_0["fixed_costs"]: parameter $f_i$
- file_0_view_0["customer"]: supermarket index $J$
- file_0_view_0["demand"]: parameter $d_j$
- file_2_view_0["Unnamed: 0"]: supplier index $I$ (rows)
- file_2_view_0["C1", "C2"]: parameter $c_{ij}$ (columns indexed by $J$)