**Abstract Mathematical Model**

**Index Sets**
- $S$: set of suppliers (from `supply_capacity.csv`, column `Unnamed: 0`)
- $C$: set of customer groups (from `customer_demand.csv`, column `customer`)

**Parameters**
- $a_s$: supply capacity of supplier $s$ (from `supply_capacity.csv`, column `supply_capacity`)
- $d_c$: demand of customer group $c$ (from `customer_demand.csv`, column `demand`)
- $t_{sc}$: transportation cost per unit from supplier $s$ to customer $c$ (from `transportation_costs.csv`, row `Unnamed: 0` for $s$, column $c$ for $c$)

**Decision Variables**
- $x_{sc} \geq 0$: number of units transported from supplier $s$ to customer $c$ (continuous, nonnegative)

**Objective**
\[
\min \sum_{s \in S} \sum_{c \in C} t_{sc} \, x_{sc}
\]

**Constraints**
1. **Supply Capacity (for each supplier):**
   \[
   \sum_{c \in C} x_{sc} \leq a_s \qquad \forall s \in S
   \]
2. **Demand Satisfaction (for each customer group):**
   \[
   \sum_{s \in S} x_{sc} = d_c \qquad \forall c \in C
   \]
3. **Nonnegativity:**
   \[
   x_{sc} \geq 0 \qquad \forall s \in S,\, c \in C
   \]

---

**Data Mapping**

- $S$: All values in `supply_capacity.csv`, column `Unnamed: 0`
- $C$: All values in `customer_demand.csv`, column `customer`
- $a_s$: `supply_capacity.csv`, columns: `Unnamed: 0` (supplier), `supply_capacity`
- $d_c$: `customer_demand.csv`, columns: `customer`, `demand`
- $t_{sc}$: `transportation_costs.csv`, row `Unnamed: 0` (supplier), columns $C$ (customer group IDs)

---

**Notes**
- All index sets, parameters, and variables are defined directly from the supplied data.
- The model ensures all customer demands are met, no supplier exceeds its capacity, and total transportation cost is minimized.