## Symbolic Mathematical Model

### Sets
- $I$: set of item options (all item_ref in item options tables, both authorized and unauthorized)
- $I_{auth} \subseteq I$: set of authorized item options (authorized=1)
- $C$: set of categories (from category limits table)
- $L$: set of storage areas/resources (from resource capacity tables, e.g., AREA_A, AREA_B, AREA_C)
- $B$: set of bundle bonus pairs (from bundle bonuses table)
- $INC$: set of incompatible pairs (from incompatible pairs table)
- $REQ$: set of requires pairs (from requires pairs table)

### Parameters
- $unit\_benefit_i$: unit_benefit_cents for item $i$ (from item options tables)
- $item\_fee_i$: item_fee_cents for item $i$ (from item options tables)
- $cat_i$: category of item $i$ (from item options tables)
- $loc_i$: location_id of item $i$ (from item options tables)
- $minlot_i$: minimum_lot for item $i$ (from item options tables)
- $maxord_i$: maximum_order for item $i$ (from item options tables)
- $auth_i$: authorized flag for item $i$ (from item options tables)
- $usage_{i\ell}$: amount of resource $\ell$ used per unit of item $i$ (from item usage tables; if missing, 0)
- $cap_\ell$: total available capacity for resource $\ell$ (sum of capacity_ledger entries for resource $\ell$)
- $catmin_c$, $catmax_c$: minimum_quantity, maximum_quantity for category $c$ (from category limits table)
- $catfee_c$: activation_fee_cents for category $c$ (from category limits table)
- $bonus_{ij}$: bonus_cents for bundle $(i,j)\in B$ (from bundle bonuses table)

### Decision Variables
- $x_i \in \mathbb{Z}_+$: integer quantity of item $i$ to select (for $i\in I_{auth}$), $x_i=0$ for $i\notin I_{auth}$
- $y_i \in \{0,1\}$: 1 if $x_i>0$, 0 otherwise (for all $i\in I$)
- $z_c \in \{0,1\}$: 1 if any item in category $c$ is selected, 0 otherwise
- $w_{ij} \in \{0,1\}$: 1 if both $y_i=1$ and $y_j=1$ for bundle $(i,j)\in B$, 0 otherwise

### Objective
Maximize net benefit in USD cents:
\[
\max \left\{
\sum_{i\in I_{auth}} \left[ unit\_benefit_i \cdot x_i - item\_fee_i \cdot y_i \right]
- \sum_{c\in C} catfee_c \cdot z_c
+ \sum_{(i,j)\in B} bonus_{ij} \cdot w_{ij}
\right\}
\]

### Constraints

#### 1. Authorization and Integer Bounds
\[
x_i = 0 \quad \forall i \notin I_{auth}
\]
\[
x_i \in \{0\} \cup [minlot_i, maxord_i] \cap \mathbb{Z} \quad \forall i \in I_{auth}
\]
\[
y_i = \begin{cases}
1 & \text{if } x_i \geq 1 \\
0 & \text{if } x_i = 0
\end{cases} \quad \forall i \in I
\]
(Enforced via: $x_i \leq maxord_i \cdot y_i$, $x_i \geq minlot_i \cdot y_i$, $x_i \geq 0$)

#### 2. Resource/Area Capacity (no borrowing)
\[
\sum_{i\in I} usage_{i\ell} \cdot x_i \leq cap_\ell \quad \forall \ell \in L
\]

#### 3. Category Quantity Limits and Activation
\[
catmin_c \leq \sum_{i:cat_i=c} x_i \leq catmax_c \quad \forall c \in C
\]
\[
z_c \geq y_i \quad \forall i: cat_i = c
\]

#### 4. Incompatibility
\[
y_i + y_j \leq 1 \quad \forall (i,j) \in INC
\]

#### 5. Requires
\[
y_i \leq y_{prereq} \quad \forall (i,prereq) \in REQ
\]

#### 6. Bundle Bonuses
\[
w_{ij} \leq y_i, \quad w_{ij} \leq y_j, \quad w_{ij} \geq y_i + y_j - 1 \quad \forall (i,j) \in B
\]

#### 7. Nonnegativity and Integrality
\[
x_i \in \mathbb{Z}_+, \quad y_i \in \{0,1\}, \quad z_c \in \{0,1\}, \quad w_{ij} \in \{0,1\}
\]

---

## Data Mapping

- Bundle bonuses: file_0_view_0, columns item_a, item_b, bonus_cents $\to$ $B$, $bonus_{ij}$
- Resource capacity: file_1_view_0, columns resource, entry, amount $\to$ $cap_\ell$ (sum by resource)
- Category limits: file_2_view_0, columns category, minimum_quantity, maximum_quantity, activation_fee_cents $\to$ $C$, $catmin_c$, $catmax_c$, $catfee_c$
- Incompatible pairs: file_4_view_0, columns item_a, item_b $\to$ $INC$
- Item options: file_5_view_0 and file_6_view_0, columns item_ref, authorized, category, minimum_lot, maximum_order, location_id, unit_benefit_cents, item_fee_cents $\to$ $I$, $I_{auth}$, $cat_i$, $loc_i$, $minlot_i$, $maxord_i$, $unit\_benefit_i$, $item\_fee_i$
- Requires pairs: file_8_view_0, columns item_ref, prerequisite_ref $\to$ $REQ$
- Item usage: file_9_view_0 and file_10_view_0, columns item_ref, resource, amount $\to$ $usage_{i\ell}$
- All variables and constraints are indexed over the union of these sets as described above.

---

**All parameters, sets, and constraints are mapped directly from the supplied tables as described.**