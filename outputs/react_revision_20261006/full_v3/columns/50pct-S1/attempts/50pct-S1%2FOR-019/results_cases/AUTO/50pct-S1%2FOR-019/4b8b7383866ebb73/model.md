##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from supplier $i$ to customer $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I = \{$supplier1, supplier2, supplier3, supplier4, supplier5, supplier6, supplier7, supplier8$\}$ (distribution centers, from supply_capacity.csv and transportation_costs.csv row axis)
- $J = \{$demand1, demand2, demand3, demand4, demand5, demand6, demand7, demand8$\}$ (customer groups, from customer_demand.csv and transportation_costs.csv column axis)

##### Parameters

- $d_j$: demand of customer $j$ (from customer_demand.csv, column "demand", table_id: file_0_view_0)
- $s_i$: supply capacity of supplier $i$ (from supply_capacity.csv, column "supply_capacity", table_id: file_1_view_0)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from transportation_costs.csv, columns "transportation_cost_to_demand1" ... "transportation_cost_to_demand8", table_id: file_2_view_0)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction** (each customer group receives at least its demand):
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
2. **Supply capacity** (each supplier does not exceed its capacity):
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
3. **Non-negativity**:
   $$
   x_{ij} \geq 0 \quad \forall i \in I, j \in J
   $$

##### Data Mapping

- $I$ (suppliers): supplier_id from supply_capacity.csv (table_id: file_1_view_0) and row axis of transportation_costs.csv (table_id: file_2_view_0)
- $J$ (customers): customer_id from customer_demand.csv (table_id: file_0_view_0) and column axis of transportation_costs.csv (table_id: file_2_view_0)
- $d_j$: demand from customer_demand.csv, column "demand", table_id: file_0_view_0, indexed by customer_id
- $s_i$: supply_capacity from supply_capacity.csv, column "supply_capacity", table_id: file_1_view_0, indexed by supplier_id
- $c_{ij}$: transportation_costs.csv, table_id: file_2_view_0, row "supplier_id" (supplier), columns "transportation_cost_to_demand1" ... "transportation_cost_to_demand8" (customer), with mapping as per relationships in the Observation

All indices, parameters, and coefficients are bound to the retrieved data as described above.