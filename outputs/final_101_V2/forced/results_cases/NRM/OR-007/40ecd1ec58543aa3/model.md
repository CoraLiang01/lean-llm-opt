#### Abstract Mathematical Model

**Index Sets**
- $S$: set of warehouses (indexed by $s$), from column "region" in table_id file_1_view_0 and "Unnamed: 0" in file_2_view_0
- $D$: set of stores (indexed by $d$), from column "customer" in table_id file_0_view_0 and columns "D1", "D2", ... in file_2_view_0

**Parameters**
- $a_s$: supply capacity of warehouse $s \in S$ (from column "supply_capacity" in table_id file_1_view_0)
- $b_d$: daily demand of store $d \in D$ (from column "demand" in table_id file_0_view_0)
- $c_{sd}$: unit transportation cost from warehouse $s$ to store $d$ (from table_id file_2_view_0, row "Unnamed: 0" = $s$, column $d$)

**Decision Variables**
- $x_{sd} \geq 0$: quantity shipped from warehouse $s$ to store $d$ (continuous, non-negative)

**Objective**
\[
\min \sum_{s \in S} \sum_{d \in D} c_{sd} \, x_{sd}
\]

**Constraints**
1. **Warehouse Supply Capacity:**
   \[
   \sum_{d \in D} x_{sd} \leq a_s \qquad \forall s \in S
   \]
2. **Store Demand Satisfaction:**
   \[
   \sum_{s \in S} x_{sd} = b_d \qquad \forall d \in D
   \]
3. **Non-negativity:**
   \[
   x_{sd} \geq 0 \qquad \forall s \in S,\, d \in D
   \]

---

**Data Mapping**

- Table file_0_view_0 ("customer_demand.csv"):
    - Index set $D$ from column "customer"
    - Parameter $b_d$ from column "demand"
- Table file_1_view_0 ("supply_capacity.csv"):
    - Index set $S$ from column "region"
    - Parameter $a_s$ from column "supply_capacity"
- Table file_2_view_0 ("transportation_costs.csv"):
    - Index set $S$ from column "Unnamed: 0"
    - Index set $D$ from columns "D1", "D2", ..., matching "customer"
    - Parameter $c_{sd}$ from cell at row "Unnamed: 0" = $s$, column $d$