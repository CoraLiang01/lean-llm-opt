##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity of goods shipped from distribution center (supplier) $i \in I$ to customer group $j \in J$.

##### Sets

- $I = \{$S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17, S18$\}$ (distribution centers)
- $J = \{$C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15, C16, C17, C18$\}$ (customer groups)

##### Parameters

- $d_j$: demand (units) for customer group $j \in J$ (from file_0_view_0, column demand_units, indexed by customer_id)
- $s_i$: supply capacity (units) for distribution center $i \in I$ (from file_1_view_0, column supply_capacity_units, indexed by supplier_id)
- $c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$ (from file_2_view_0, column transportation_cost_to_Ck, row supplier_id)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   For each customer group $j \in J$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
2. **Supply capacity:**  
   For each distribution center $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

##### Data Mapping

- $I$ (distribution centers): supplier_id from file_1_view_0 and file_2_view_0
- $J$ (customer groups): customer_id from file_0_view_0 and columns transportation_cost_to_Ck in file_2_view_0
- $d_j$: file_0_view_0, column demand_units, indexed by customer_id
- $s_i$: file_1_view_0, column supply_capacity_units, indexed by supplier_id
- $c_{ij}$: file_2_view_0, row supplier_id $i$, column transportation_cost_to_Ck for customer $j$