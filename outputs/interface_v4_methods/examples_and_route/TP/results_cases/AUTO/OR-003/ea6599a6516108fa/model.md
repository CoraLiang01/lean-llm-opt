#### Abstract Mathematical Model

**Index Sets:**
- $S$: set of suppliers (from `supply_capacity.csv`, column `Unnamed: 0`)
- $C$: set of customer groups (from `customer_demand.csv`, column `customer`)

**Parameters:**
- $a_s$: supply capacity of supplier $s \in S$ (from `supply_capacity.csv`, column `supply_capacity`)
- $d_c$: demand of customer group $c \in C$ (from `customer_demand.csv`, column `demand`)
- $t_{sc}$: transportation cost per unit from supplier $s$ to customer $c$ (from `transportation_costs.csv`, row `Unnamed: 0` and column $c$)

**Decision Variables:**
- $x_{sc} \geq 0$: number of units transported from supplier $s$ to customer $c$ (continuous, unless otherwise specified)

**Objective:**
\[
\min \sum_{s \in S} \sum_{c \in C} t_{sc} \cdot x_{sc}
\]

**Constraints:**
1. **Supply Capacity (for each supplier):**
   \[
   \sum_{c \in C} x_{sc} \leq a_s, \quad \forall s \in S
   \]
2. **Demand Satisfaction (for each customer group):**
   \[
   \sum_{s \in S} x_{sc} = d_c, \quad \forall c \in C
   \]
3. **Nonnegativity:**
   \[
   x_{sc} \geq 0, \quad \forall s \in S,\, c \in C
   \]

---

#### Data Mapping

- $S$: All values in `supply_capacity.csv`, column `Unnamed: 0`
- $C$: All values in `customer_demand.csv`, column `customer`
- $a_s$: `supply_capacity.csv`, columns: `Unnamed: 0` (supplier), `supply_capacity`
- $d_c$: `customer_demand.csv`, columns: `customer`, `demand`
- $t_{sc}$: `transportation_costs.csv`, rows: `Unnamed: 0` (supplier), columns: $C$ (customer group IDs)

All index sets, parameters, and constraints are mapped directly to the original file columns and business identifiers as required.