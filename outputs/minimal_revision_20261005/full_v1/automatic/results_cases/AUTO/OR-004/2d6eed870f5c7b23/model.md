##### Symbolic Model

Let $I = \{S1, S2, \ldots, S12\}$ (distribution centers), $J = \{C1, C2, \ldots, C12\}$ (customer groups).

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from distribution center $i \in I$ to customer group $j \in J$ (continuous).

Parameters:
- $d_j$: demand of customer group $j$ (from "customer_demand.csv")
- $s_i$: supply capacity of distribution center $i$ (from "supply_capacity.csv")
- $c_{ij}$: transportation cost per unit from $i$ to $j$ (from "transportation_costs.csv")

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
\[
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
\]
\[
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
\]
\[
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\]

---

##### Data Mapping

- $d_j$ (customer demand): from "customer_demand.csv", table_id: file_0_view_0, columns: customer, demand, rows: 0–11
- $s_i$ (supply capacity): from "supply_capacity.csv", table_id: file_1_view_0, columns: Unnamed: 0, supply_capacity, rows: 0–11
- $c_{ij}$ (transportation cost): from "transportation_costs.csv", table_id: file_2_view_0, columns: Unnamed: 0 (rows: S1–S12), C1–C12, rows: 0–11

All identifiers and coefficients are to be used exactly as in the retrieved tables, preserving source order.