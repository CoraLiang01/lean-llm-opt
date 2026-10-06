#### Abstract Mathematical Model

**Sets:**
- $S$: Set of distribution centers (indexed by $s$), from `supply_capacity.csv`, column `Unnamed: 0`
- $C$: Set of customer groups (indexed by $c$), from `customer_demand.csv`, column `customer`

**Parameters:**
- $d_c$: Demand of customer group $c$, from `customer_demand.csv`, column `demand`
- $u_s$: Supply capacity of distribution center $s$, from `supply_capacity.csv`, column `supply_capacity`
- $t_{sc}$: Transportation cost per unit from distribution center $s$ to customer group $c$, from `transportation_costs.csv`, row `Unnamed: 0` (distribution center), column $c$ (customer)

**Decision Variables:**
- $x_{sc} \in \mathbb{R}_{\geq 0}$: Number of units shipped from distribution center $s$ to customer group $c$

**Objective:**
\[
\min \sum_{s \in S} \sum_{c \in C} t_{sc} \, x_{sc}
\]

**Constraints:**

1. **Demand Satisfaction:**
   \[
   \sum_{s \in S} x_{sc} = d_c, \quad \forall c \in C
   \]

2. **Supply Capacity:**
   \[
   \sum_{c \in C} x_{sc} \leq u_s, \quad \forall s \in S
   \]

3. **Nonnegativity:**
   \[
   x_{sc} \geq 0, \quad \forall s \in S,\, c \in C
   \]

---

#### Data Mapping

- $S$: All values in `supply_capacity.csv`, column `Unnamed: 0`
- $C$: All values in `customer_demand.csv`, column `customer`
- $d_c$: `customer_demand.csv`, column `demand`, key `customer`
- $u_s$: `supply_capacity.csv`, column `supply_capacity`, key `Unnamed: 0`
- $t_{sc}$: `transportation_costs.csv`, row `Unnamed: 0` (distribution center), column $c$ (customer)

All sets, parameters, and indices are to be used exactly as they appear in the source files and columns.