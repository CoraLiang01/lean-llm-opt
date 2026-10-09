##### Decision Variables

For each distribution center $i \in I$ and customer group $j \in J$:
- $x_{ij} \geq 0$: quantity shipped from distribution center $i$ to customer group $j$ (continuous).

##### Parameters

- $I$: set of distribution centers (from "supply_capacity.csv"): $\{S1, S2, \ldots, S18\}$
- $J$: set of customer groups (from "customer_demand.csv"): $\{C1, C2, \ldots, C18\}$
- $d_j$: demand of customer group $j$ (from "customer_demand.csv")
- $s_i$: supply capacity of distribution center $i$ (from "supply_capacity.csv")
- $c_{ij}$: transportation cost per unit from $i$ to $j$ (from "transportation_costs.csv")

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction** (each customer group receives at least its demand):

   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]

2. **Supply capacity** (each distribution center does not exceed its capacity):

   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]

3. **Non-negativity**:

   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

---

##### Data Mapping

- $I$ (distribution centers): all "supplier_id" in table_id="file_1_view_0" (supply_capacity.csv)
- $J$ (customer groups): all "customer_id" in table_id="file_0_view_0" (customer_demand.csv)
- $d_j$: "demand_units" for customer $j$ in table_id="file_0_view_0"
- $s_i$: "supply_capacity_units" for supplier $i$ in table_id="file_1_view_0"
- $c_{ij}$: "transportation_cost_to_$j$" for supplier $i$ in table_id="file_2_view_0", with $j$ as in "customer_id" from table_id="file_0_view_0"

All index sets, parameters, and coefficients are defined by the full set of records in the respective CSV files as returned above.