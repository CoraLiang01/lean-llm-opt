## Symbolic Mathematical Model

### Sets
- $I$: set of authorized item options (from file_2_view_0, column item_ref)
- $C$: set of categories (from file_4_view_0, column category)
- $R$: set of resources (from file_8_view_0, column resource)
- $B$: set of bundle pairs (from file_7_view_0, columns item_a, item_b)
- $P$: set of prerequisite pairs (from file_0_view_0, columns item_ref, prerequisite_ref)
- $Q$: set of incompatible pairs (from file_5_view_0, columns item_a, item_b)

### Parameters
- $b_i$: unit benefit in cents for item $i$ (file_2_view_0, unit_benefit_cents)
- $f_i$: fixed fee in cents for item $i$ (file_2_view_0, item_fee_cents)
- $l_i$: minimum lot for item $i$ (file_2_view_0, minimum_lot)
- $u_i$: maximum order for item $i$ (file_2_view_0, maximum_order)
- $cat(i)$: category of item $i$ (file_2_view_0, category)
- $a_{i,r}$: per-unit usage of resource $r$ by item $i$ (file_3_view_0, amount; unit conversion applied)
- $q_c^{\min}$: minimum quantity for category $c$ (file_4_view_0, minimum_quantity)
- $q_c^{\max}$: maximum quantity for category $c$ (file_4_view_0, maximum_quantity)
- $F_c$: activation fee for category $c$ (file_4_view_0, activation_fee_cents)
- $cap_r$: available capacity for resource $r$ (sum of file_8_view_0, amount, for each resource $r$; unit conversion applied)
- $bonus_{b}$: bonus in cents for bundle $b$ (file_7_view_0, bonus_cents)
- $A_b, B_b$: items in bundle $b$ (file_7_view_0, item_a, item_b)
- $prereq(i)$: set of prerequisite items for $i$ (file_0_view_0)
- $inc(i)$: set of items incompatible with $i$ (file_5_view_0)

### Decision Variables
- $x_i \in \mathbb{Z}_+$: quantity ordered of item $i \in I$
- $z_i \in \{0,1\}$: 1 if $x_i > 0$, 0 otherwise, for $i \in I$
- $y_c \in \{0,1\}$: 1 if any item in category $c$ is ordered, 0 otherwise
- $w_b \in \{0,1\}$: 1 if both items in bundle $b$ are ordered, 0 otherwise

### Objective
Maximize net benefit in cents:
\[
\max \left\{
\sum_{i \in I} b_i x_i
- \sum_{i \in I} f_i z_i
- \sum_{c \in C} F_c y_c
+ \sum_{b \in B} bonus_b w_b
\right\}
\]

### Constraints

#### Item order bounds
\[
\forall i \in I: \quad x_i = 0 \quad \text{or} \quad l_i \leq x_i \leq u_i
\]
\[
\forall i \in I: \quad z_i = 1 \iff x_i \geq l_i
\]
\[
\forall i \in I: \quad x_i \leq u_i z_i
\]
\[
\forall i \in I: \quad x_i \geq l_i z_i
\]
\[
\forall i \in I: \quad x_i \in \mathbb{Z}_+, \quad z_i \in \{0,1\}
\]

#### Category activation and quantity bounds
\[
\forall c \in C: \quad y_c \geq z_i \quad \forall i \in I: cat(i) = c
\]
\[
\forall c \in C: \quad q_c^{\min} y_c \leq \sum_{i: cat(i) = c} x_i \leq q_c^{\max} y_c
\]
\[
\forall c \in C: \quad y_c \in \{0,1\}
\]

#### Resource constraints (with unit conversion)
Let $a_{i,r}^*$ be $a_{i,r}$ converted to the unit of $cap_r$:
- For power: $a_{i,r}$ in kwh $\to$ $a_{i,r}^* = 1000 \cdot a_{i,r}$ wh
- For space: $a_{i,r}$ in liter $\to$ $a_{i,r}^* = 1000 \cdot a_{i,r}$ ml
- For labor: $a_{i,r}$ in hour $\to$ $a_{i,r}^* = 60 \cdot a_{i,r}$ minute

\[
\forall r \in R: \quad \sum_{i \in I} a_{i,r}^* x_i \leq cap_r
\]

#### Incompatibility
\[
\forall (i,j) \in Q: \quad z_i + z_j \leq 1
\]

#### Prerequisites
\[
\forall (i, p) \in P: \quad z_i \leq z_p
\]

#### Bundle bonuses
\[
\forall b \in B: \quad w_b \leq z_{A_b}, \quad w_b \leq z_{B_b}, \quad w_b \geq z_{A_b} + z_{B_b} - 1
\]
\[
\forall b \in B: \quad w_b \in \{0,1\}
\]

#### Only authorized items
\[
I = \{i: \text{authorized} = 1 \text{ in file_2_view_0}\}
\]

### Data Mapping

- Items, categories, and all item parameters: file_2_view_0 (columns: item_ref, category, authorized, minimum_lot, maximum_order, unit_benefit_cents, item_fee_cents)
- Resource usage: file_3_view_0 (columns: item_ref, resource, amount, unit)
- Resource capacities: file_8_view_0 (columns: resource, entry, amount, unit; sum by resource, apply sign, convert units)
- Category constraints: file_4_view_0 (columns: category, minimum_quantity, maximum_quantity, activation_fee_cents)
- Prerequisite pairs: file_0_view_0 (columns: item_ref, prerequisite_ref)
- Incompatible pairs: file_5_view_0 (columns: item_a, item_b)
- Bundle bonuses: file_7_view_0 (columns: item_a, item_b, bonus_cents)

All sets, parameters, and constraints are defined directly from these tables, using only rows with authorized=1 for items. All units are converted as specified. The objective and all constraints are in USD cents.