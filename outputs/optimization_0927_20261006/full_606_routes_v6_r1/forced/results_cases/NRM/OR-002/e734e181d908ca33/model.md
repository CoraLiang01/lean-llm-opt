#### Abstract Mathematical Model

**Index Sets:**
- $S$: Set of Walmart store identifiers (from `supply_capacity.csv`, column `Unnamed: 0`)
- $C$: Set of customer group identifiers (from `customer_demand.csv`, column `customer`)

**Parameters:**
- $d_c$: Demand of customer group $c \in C$ (from `customer_demand.csv`, column `demand`)
- $u_s$: Supply capacity of Walmart store $s \in S$ (from `supply_capacity.csv`, column `supply_capacity`)
- $t_{s,c}$: Transportation cost per unit from Walmart store $s$ to customer group $c$ (from `transportation_costs.csv`, columns `Unnamed: 0` for $s$ and $c$ for customer group)

**Decision Variables:**
- $x_{s,c} \geq 0$: Quantity of goods transported from Walmart store $s \in S$ to customer group $c \in C$

**Objective:**
\[
\min \sum_{s \in S} \sum_{c \in C} t_{s,c} \cdot x_{s,c}
\]

**Constraints:**

1. **Demand Satisfaction:**
   \[
   \sum_{s \in S} x_{s,c} = d_c, \quad \forall c \in C
   \]

2. **Supply Capacity:**
   \[
   \sum_{c \in C} x_{s,c} \leq u_s, \quad \forall s \in S
   \]

3. **Non-negativity:**
   \[
   x_{s,c} \geq 0, \quad \forall s \in S,\, c \in C
   \]

---

#### Data Mapping

- $S$ (Walmart store identifiers): `supply_capacity.csv`, column `Unnamed: 0`
- $C$ (customer group identifiers): `customer_demand.csv`, column `customer`
- $d_c$: `customer_demand.csv`, column `demand`
- $u_s$: `supply_capacity.csv`, column `supply_capacity`
- $t_{s,c}$: `transportation_costs.csv`, row `Unnamed: 0` (store), column `$c$` (customer group)