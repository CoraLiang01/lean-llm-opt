## Symbolic Mathematical Model

### Sets
- $I$: set of all item options (all item_ref in both item tables, with authorized status and all attributes)
- $I_{auth} \subseteq I$: set of authorized items ($\text{authorized}=1$)
- $C$: set of all categories (from category limits table)
- $R$: set of all resources (from resource capacity table)
- $B$: set of all bundle bonus pairs (from bundle table, as ordered pairs $(a,b)$)
- $Q$: set of all incompatibility pairs (from incompatibility table, as unordered pairs $\{a,b\}$)
- $P$: set of all prerequisite pairs (from prerequisite table, as ordered pairs $(i,p)$: $i$ requires $p$)

### Parameters
- $\text{unit\_benefit}_i$: unit benefit in cents for item $i$ (from item tables)
- $\text{item\_fee}_i$: fixed fee in cents for item $i$ if any are selected (from item tables)
- $\text{min\_lot}_i$: minimum lot size for item $i$ (from item tables)
- $\text{max\_order}_i$: maximum order for item $i$ (from item tables)
- $\text{cat}_i$: category of item $i$ (from item tables)
- $\text{loc}_i$: location (resource) of item $i$ (from item tables)
- $\text{usage}_{i,r}$: per-unit usage of resource $r$ by item $i$ (from item usage tables; zero if not listed)
- $\text{cap}_r$: total available capacity for resource $r$ (sum of all capacity_ledger entries for $r$)
- $\text{cat\_min}_c$, $\text{cat\_max}_c$: min/max total quantity for category $c$ (from category limits table)
- $\text{cat\_fee}_c$: activation fee for category $c$ (from category limits table)
- $\text{bonus}_{a,b}$: bundle bonus in cents for pair $(a,b)$ (from bundle table)
- $I_c$: set of items in category $c$
- $I_r$: set of items using resource $r$

### Decision Variables
- $x_i \in \mathbb{Z}_{\geq 0}$: quantity of item $i$ selected (integer)
- $y_i \in \{0,1\}$: 1 if any of item $i$ is selected, 0 otherwise
- $z_c \in \{0,1\}$: 1 if any item in category $c$ is selected, 0 otherwise
- $w_{a,b} \in \{0,1\}$: 1 if both $y_a=1$ and $y_b=1$ (for bundle $(a,b)$), 0 otherwise

### Objective
Maximize net benefit in cents:
\[
\max \left\{
\sum_{i \in I_{auth}} \text{unit\_benefit}_i x_i
- \sum_{i \in I_{auth}} \text{item\_fee}_i y_i
- \sum_{c \in C} \text{cat\_fee}_c z_c
+ \sum_{(a,b) \in B} \text{bonus}_{a,b} w_{a,b}
\right\}
\]

### Constraints

#### 1. Item selection and bounds
For all $i \in I$:
\[
x_i = 0 \quad \text{if $i \notin I_{auth}$}
\]
For all $i \in I_{auth}$:
\[
y_i \in \{0,1\}
\]
\[
x_i \geq 0, \quad x_i \in \mathbb{Z}
\]
\[
x_i \leq \text{max\_order}_i \cdot y_i
\]
\[
x_i \geq \text{min\_lot}_i \cdot y_i
\]
\[
x_i = 0 \text{ if not authorized}
\]

#### 2. Resource capacity (per resource, no borrowing)
For all $r \in R$:
\[
\sum_{i \in I_{auth}: \text{loc}_i = r} \text{usage}_{i,r} x_i \leq \text{cap}_r
\]

#### 3. Category quantity limits and activation
For all $c \in C$:
\[
z_c \in \{0,1\}
\]
\[
\sum_{i \in I_{auth}: \text{cat}_i = c} x_i \geq \text{cat\_min}_c \cdot z_c
\]
\[
\sum_{i \in I_{auth}: \text{cat}_i = c} x_i \leq \text{cat\_max}_c \cdot z_c
\]
\[
z_c \geq y_i \quad \forall i \in I_{auth}: \text{cat}_i = c
\]

#### 4. Incompatibility
For all $\{a,b\} \in Q$:
\[
y_a + y_b \leq 1
\]

#### 5. Prerequisite
For all $(i,p) \in P$:
\[
y_i \leq y_p
\]

#### 6. Bundle bonuses
For all $(a,b) \in B$:
\[
w_{a,b} \leq y_a
\]
\[
w_{a,b} \leq y_b
\]
\[
w_{a,b} \geq y_a + y_b - 1
\]
\[
w_{a,b} \in \{0,1\}
\]

#### 7. Zero for unauthorized
For all $i \notin I_{auth}$:
\[
x_i = 0, \quad y_i = 0
\]

### Data Mapping

- Bundle bonus: table_id file_0_view_0, columns item_a, item_b, bonus_cents
- Resource capacity: table_id file_1_view_0, columns resource, entry, amount, unit (sum by resource)
- Category limits: table_id file_2_view_0, columns category, minimum_quantity, maximum_quantity, activation_fee_cents
- Incompatibility: table_id file_4_view_0, columns item_a, item_b
- Item options: table_id file_5_view_0 and file_6_view_0, columns item_ref, category, authorized, minimum_lot, maximum_order, configuration_id, location_id, unit_benefit_cents, item_fee_cents
- Prerequisite: table_id file_8_view_0, columns item_ref, prerequisite_ref
- Item usage: table_id file_9_view_0 and file_10_view_0, columns item_ref, resource, amount, unit

All sets, parameters, and constraints are defined directly from these tables, using all rows as described above.