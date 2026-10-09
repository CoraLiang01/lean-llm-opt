#### Mathematical Model

Let $I$ be the set of distribution centers (from supplier_id in supply_capacity.csv and transportation_costs.csv), and $J$ the set of customer groups (from customer_id in customer_demand.csv and transportation_costs.csv columns).

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from distribution center $i \in I$ to customer group $j \in J$ (continuous).

Objective:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:
- Demand satisfaction:
$$
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
$$
- Supply capacity:
$$
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
$$
- Nonnegativity:
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

#### Data Mapping

- $I$: All supplier_id in supply_capacity.csv (file_1_view_0) and transportation_costs.csv (file_2_view_0) rows: supply1, supply2, ..., supply8.
- $J$: All customer_id in customer_demand.csv (file_0_view_0) and transportation_costs.csv columns: demand1, demand2, ..., demand8.
- $d_j$: demand for customer group $j$ from column demand in customer_demand.csv (file_0_view_0), indexed by customer_id.
- $s_i$: supply_capacity for distribution center $i$ from supply_capacity.csv (file_1_view_0), indexed by supplier_id.
- $c_{ij}$: transportation_cost_to_demandX for each $i$ (supplier_id) and $j$ (demandX) from transportation_costs.csv (file_2_view_0).

Index sets and parameter mappings are defined by the exact identifiers and columns as above. No data is omitted or aggregated.