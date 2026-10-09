## Symbolic Mathematical Model

### Sets
- $I$: set of items (indexed by $i$)
- $O$: set of options (indexed by $o$), each corresponding to an item $i(o)$
- $C$: set of categories (indexed by $c$)
- $R$: set of resources (indexed by $r$)
- $B$: set of bundle pairs $(o_1, o_2)$ eligible for a bundle bonus
- $P$: set of incompatible option pairs $(o, o')$
- $Q$: set of requires pairs $(o, o')$ where $o$ requires $o'$

### Parameters (all in USD cents unless otherwise noted)
- $\text{benefit}_i$: per-unit net benefit for item $i$ (sum of all signed amount_cents for $i$)
- $\text{item\_fee}_o$: fixed fee for option $o$ if any units are ordered (from item_fee table, by item_ref)
- $\text{min\_lot}_o$, $\text{max\_order}_o$: minimum and maximum integer lot size for option $o$ (from item table)
- $\text{authorized}_o \in \{0,1\}$: 1 if option $o$ is authorized, 0 otherwise (from item table)
- $\text{cat}_o$: category of option $o$ (from item table)
- $\text{cat\_min}_c$, $\text{cat\_max}_c$: minimum and maximum total quantity for category $c$ (from category table)
- $\text{cat\_fee}_c$: activation fee for category $c$ (from category table)
- $\text{usage}_{o,r}$: resource $r$ usage per unit of option $o$ (from usage table, with unit conversion)
- $\text{cap}_r$: available capacity for resource $r$ (from capacity_ledger table, with unit conversion)
- $\text{bundle\_bonus}_{(o_1,o_2)}$: bonus for bundle $(o_1,o_2)$ (from bundle table)
- $P$: set of unordered incompatible pairs $(o,o')$
- $Q$: set of requires pairs $(o,o')$ (option $o$ requires $o'$)

### Decision Variables
- $x_o \in \mathbb{Z}_{\geq 0}$: integer quantity ordered for option $o$
- $y_o \in \{0,1\}$: 1 if $x_o > 0$, 0 otherwise (option $o$ is used)
- $z_c \in \{0,1\}$: 1 if any $x_o > 0$ for $o$ in category $c$ (category $c$ is used)
- $b_{(o_1,o_2)} \in \{0,1\}$: 1 if both $x_{o_1} > 0$ and $x_{o_2} > 0$, 0 otherwise (bundle bonus awarded)

### Objective
Maximize net benefit:
\[
\max \left\{
\sum_{o \in O} \text{benefit}_{i(o)} x_o
- \sum_{o \in O} \text{item\_fee}_o y_o
- \sum_{c \in C} \text{cat\_fee}_c z_c
+ \sum_{(o_1,o_2) \in B} \text{bundle\_bonus}_{(o_1,o_2)} b_{(o_1,o_2)}
\right\}
\]

### Constraints

#### Option authorization and lot size
\[
x_o = 0 \quad \forall o: \text{authorized}_o = 0
\]
\[
y_o \in \{0,1\} \quad \forall o
\]
\[
x_o \geq \text{min\_lot}_o \cdot y_o \quad \forall o: \text{authorized}_o = 1
\]
\[
x_o \leq \text{max\_order}_o \cdot y_o \quad \forall o: \text{authorized}_o = 1
\]

#### Category activation and quantity limits
\[
z_c \geq y_o \quad \forall o: \cat_o = c
\]
\[
\sum_{o: \cat_o = c} x_o \geq \text{cat\_min}_c \cdot z_c \quad \forall c
\]
\[
\sum_{o: \cat_o = c} x_o \leq \text{cat\_max}_c \quad \forall c
\]

#### Resource capacity (with unit conversion)
\[
\sum_{o \in O} \text{usage}_{o,r} x_o \leq \text{cap}_r \quad \forall r
\]
(Apply: 1000 ml = 1 l, 60 min = 1 h, 1000 wh = 1 kwh as needed for unit consistency.)

#### Incompatibility
\[
y_o + y_{o'} \leq 1 \quad \forall (o,o') \in P
\]

#### Requires dependencies
\[
y_o \leq y_{o'} \quad \forall (o,o') \in Q
\]
or equivalently,
\[
x_o \leq M \cdot y_{o'} \quad \forall (o,o') \in Q
\]
where $M$ is a sufficiently large constant.

#### Bundle bonuses
\[
b_{(o_1,o_2)} \leq y_{o_1}
\]
\[
b_{(o_1,o_2)} \leq y_{o_2}
\]
\[
b_{(o_1,o_2)} \geq y_{o_1} + y_{o_2} - 1
\]
for all $(o_1,o_2) \in B$

#### Variable domains
\[
x_o \in \mathbb{Z}_{\geq 0} \quad \forall o
\]
\[
y_o \in \{0,1\} \quad \forall o
\]
\[
z_c \in \{0,1\} \quad \forall c
\]
\[
b_{(o_1,o_2)} \in \{0,1\} \quad \forall (o_1,o_2) \in B
\]

### Data Mapping

- $I$: All items with at least one authorized option after filtering for latest non-DELETE revision per (tenant, table, record_id) on or before 2026-03-12 (from item and identity tables).
- $O$: All options (item_ref) with authorized = 1 after filtering as above (from item table).
- $C$: All categories present in the filtered item and category tables.
- $R$: All resources present in usage and capacity_ledger tables.
- $B$: All bundle pairs from bundle table, using latest revision per (tenant, table, record_id).
- $P$: All incompatible pairs from incompatible table, using latest revision per (tenant, table, record_id).
- $Q$: All requires pairs from requires table, using latest revision per (tenant, table, record_id).

- $\text{benefit}_i$: For each item $i$, sum all amount_cents from benefit table (latest revision per (tenant, table, record_id)), grouped by item_ref.
- $\text{item\_fee}_o$: From item_fee table, by item_ref, latest revision.
- $\text{min\_lot}_o$, $\text{max\_order}_o$, $\text{authorized}_o$, $\text{cat}_o$: From item table, by item_ref, latest revision.
- $\text{cat\_min}_c$, $\text{cat\_max}_c$, $\text{cat\_fee}_c$: From category table, by category, latest revision.
- $\text{usage}_{o,r}$: From usage table, by item_ref and resource, latest revision, with unit conversion as specified.
- $\text{cap}_r$: For each resource $r$, sum all amount from capacity_ledger table, by resource, latest revision, with unit conversion as specified.
- $\text{bundle\_bonus}_{(o_1,o_2)}$: From bundle table, by (item_a, item_b), latest revision.
- $P$: All unordered pairs from incompatible table, latest revision.
- $Q$: All pairs from requires table, latest revision.

All filtering: For each (tenant, table, record_id), keep only the highest integer revision on or before 2026-03-12, discard if action is DELETE, and count identical retransmissions once.

### Output
Report the maximum net benefit in USD cents.

---

**All parameters, sets, and constraints are mapped exactly to the filtered, deduplicated, and joined data as described above.**