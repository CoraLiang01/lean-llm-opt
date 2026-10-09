##### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups, as defined by the data.

Let $x_{ij} \geq 0$ be the continuous quantity shipped from supplier $i \in I$ to customer $j \in J$.

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:
- Demand satisfaction for each customer group:
$$
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
$$
- Supply capacity for each distribution center:
$$
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
$$
- Non-negativity:
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

##### Data Mapping

- $I$ (Suppliers): All supplier_id in supply_capacity.csv and transportation_costs.csv (S1, S2, ..., S12)
- $J$ (Customers): All customer_id in customer_demand.csv and transportation_costs.csv (C1, C2, ..., C12)
- $d_j$: demand for customer $j$ from customer_demand.csv, column demand, table_id file_0_view_0, indexed by customer_id
- $s_i$: supply_capacity for supplier $i$ from supply_capacity.csv, column supply_capacity, table_id file_1_view_0, indexed by supplier_id
- $c_{ij}$: transportation_cost from supplier $i$ to customer $j$ from transportation_costs.csv, columns transportation_cost_to_C*, table_id file_2_view_0, indexed by supplier_id and customer_id

Index sets, parameters, and all coefficients are defined by the current CSV data as described above.