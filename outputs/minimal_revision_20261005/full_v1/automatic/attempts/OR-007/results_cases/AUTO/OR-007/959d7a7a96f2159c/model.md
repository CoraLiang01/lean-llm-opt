##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i$ to store $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$ (warehouses)
- $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$ (stores)

##### Parameters

- $d_j$: demand of store $j$ (from customer_demand.csv)
- $s_i$: supply capacity of warehouse $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from transportation_costs.csv)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. Demand satisfaction (each store receives at least its demand):
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
2. Supply capacity (each warehouse ships no more than its capacity):
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
3. Non-negativity:
   $$
   x_{ij} \geq 0 \quad \forall i \in I, j \in J
   $$

---

#### Data Mapping

- $d_j$ (store demand): from customer_demand.csv (table_id: file_0_view_0, columns: customer, demand)
- $s_i$ (warehouse supply capacity): from supply_capacity.csv (table_id: file_1_view_0, columns: region, supply_capacity)
- $c_{ij}$ (transportation cost): from transportation_costs.csv (table_id: file_2_view_0, columns: Unnamed: 0 [warehouse], D1, D2, D3, D4, D5 [stores])

All identifiers and coefficients are to be used exactly as in the retrieved tables, preserving source order.