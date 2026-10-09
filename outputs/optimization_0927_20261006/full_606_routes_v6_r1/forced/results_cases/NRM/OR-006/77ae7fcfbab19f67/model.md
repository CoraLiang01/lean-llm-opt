#### Abstract Mathematical Model

**Index Sets:**
- $S$: Set of warehouses (from `supply_capacity.csv`, column `Unnamed: 0`)
- $C$: Set of stores (from `customer_demand.csv`, column `customer`)

**Parameters:**
- $d_c$: Demand of store $c \in C$ (from `customer_demand.csv`, column `demand`)
- $u_s$: Supply capacity of warehouse $s \in S$ (from `supply_capacity.csv`, column `supply_capacity`)
- $cost_{s,c}$: Transportation cost per unit from warehouse $s$ to store $c$ (from `transportation_costs.csv`, columns `Unnamed: 0` for $s$, columns $C$ for $c$)

**Decision Variables:**
- $x_{s,c} \geq 0$: Quantity shipped from warehouse $s$ to store $c$ (continuous or integer, as appropriate)

**Objective:**
\[
\min \sum_{s \in S} \sum_{c \in C} cost_{s,c} \cdot x_{s,c}
\]

**Constraints:**

1. **Demand Satisfaction (for each store):**
   \[
   \sum_{s \in S} x_{s,c} = d_c \quad \forall c \in C
   \]

2. **Warehouse Supply Capacity (for each warehouse):**
   \[
   \sum_{c \in C} x_{s,c} \leq u_s \quad \forall s \in S
   \]

3. **Non-negativity:**
   \[
   x_{s,c} \geq 0 \quad \forall s \in S,\, c \in C
   \]

---

#### Data Mapping

- **customer_demand.csv**: 
  - Table ID: `file_0_view_0`
  - Store index set $C$ from column `customer`
  - Parameter $d_c$ from column `demand`
- **supply_capacity.csv**: 
  - Table ID: `file_1_view_0`
  - Warehouse index set $S$ from column `Unnamed: 0`
  - Parameter $u_s$ from column `supply_capacity`
- **transportation_costs.csv**: 
  - Table ID: `file_2_view_0`
  - Parameter $cost_{s,c}$ from row index `Unnamed: 0` (for $s$) and columns $C$ (for $c$)

All records from each table are included as returned by CSVQA. No additional filtering or aggregation is applied.