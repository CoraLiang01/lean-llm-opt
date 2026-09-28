#### Abstract Mathematical Model

**Index Sets:**
- $S$: Set of Walmart stores (from `supply_capacity.csv`, column `Unnamed: 0`)
- $C$: Set of customer groups (from `customer_demand.csv`, column `customer`)

**Parameters:**
- $a_s$: Supply capacity of store $s \in S$ (from `supply_capacity.csv`, column `supply_capacity`)
- $d_c$: Demand of customer group $c \in C$ (from `customer_demand.csv`, column `demand`)
- $t_{s,c}$: Transportation cost per unit from store $s$ to customer group $c$ (from `transportation_costs.csv`, columns indexed by $c$, rows indexed by $s$)

**Decision Variables:**
- $x_{s,c} \geq 0$: Quantity transported from store $s$ to customer group $c$ (continuous, non-negative)

**Objective:**
\[
\min \sum_{s \in S} \sum_{c \in C} t_{s,c} \cdot x_{s,c}
\]

**Constraints:**
1. **Supply Capacity at Each Store:**
   \[
   \sum_{c \in C} x_{s,c} \leq a_s \quad \forall s \in S
   \]
2. **Demand Satisfaction for Each Customer Group:**
   \[
   \sum_{s \in S} x_{s,c} = d_c \quad \forall c \in C
   \]
3. **Non-negativity:**
   \[
   x_{s,c} \geq 0 \quad \forall s \in S,\, c \in C
   \]

---

**Data Mapping:**

- Table `file_0_view_0` (`customer_demand.csv`):  
  - Index set $C$ from column `customer`
  - Parameter $d_c$ from column `demand`
- Table `file_1_view_0` (`supply_capacity.csv`):  
  - Index set $S$ from column `Unnamed: 0`
  - Parameter $a_s$ from column `supply_capacity`
- Table `file_2_view_0` (`transportation_costs.csv`):  
  - Parameter $t_{s,c}$ from columns indexed by $c$ (customer group) and rows indexed by $s$ (store), with row id column `Unnamed: 0` and column id `customer` (see relationships in observation)