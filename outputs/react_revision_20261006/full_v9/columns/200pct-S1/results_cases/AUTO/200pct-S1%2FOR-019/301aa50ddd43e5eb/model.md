##### Mathematical Model

Let:
- $I$ = set of distribution centers (suppliers): $\{\text{supplier1}, \text{supplier2}, \text{supplier3}, \text{supplier4}, \text{supplier5}, \text{supplier6}, \text{supplier7}, \text{supplier8}\}$
- $J$ = set of customer groups (demands): $\{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$
- $x_{ij} \geq 0$ = quantity shipped from supplier $i \in I$ to customer $j \in J$ (continuous)
- $c_{ij}$ = unit transportation cost from supplier $i$ to customer $j$
- $d_j$ = demand of customer $j$
- $s_i$ = supply capacity of supplier $i$

Objective:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:
1. Demand satisfaction:
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
2. Supply capacity:
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
3. Non-negativity:
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

##### Data Mapping

- $I$ (suppliers): All unique values in column supplier_id of file_1_view_0 (supply_capacity.csv)
- $J$ (customers): All unique values in column customer_id of file_0_view_0 (customer_demand.csv)
- $d_j$: demand for customer $j$ from column demand in file_0_view_0, indexed by customer_id
- $s_i$: supply_capacity for supplier $i$ from column supply_capacity in file_1_view_0, indexed by supplier_id
- $c_{ij}$: transportation_cost from supplier $i$ to customer $j$ from file_2_view_0 (transportation_costs.csv), with row index supplier_id and column index transportation_cost_to_{customer_id} (see relationships.matrix)
- $x_{ij}$: decision variable for each $(i,j) \in I \times J$ (continuous, nonnegative)

Index sets, parameters, and cost matrix are defined exactly as in the current CSVQA Observation, preserving all identifiers and source order.