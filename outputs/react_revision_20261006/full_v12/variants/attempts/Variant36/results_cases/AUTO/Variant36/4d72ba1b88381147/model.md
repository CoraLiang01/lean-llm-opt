## Symbolic Mathematical Model

**Sets**
- $I$: set of item options (all item_ref in item tables with authorized = 1)
- $C$: set of categories (from category constraints table)
- $R$: set of resources (from resource capacity table)
- $B$: set of bundle pairs (from bundle bonuses table)
- $Q$: set of incompatible pairs (from incompatible pairs table)
- $P$: set of requires pairs (from requires pairs table)

**Parameters**
- $unit\_benefit_i$: unit_benefit_cents for item $i$ (from item tables)
- $item\_fee_i$: item_fee_cents for item $i$ (from item tables)
- $minlot_i$, $maxorder_i$: minimum_lot, maximum_order for item $i$ (from item tables)
- $cat_i$: category of item $i$ (from item tables)
- $loc_i$: location_id of item $i$ (from item tables)
- $auth_i$: authorized flag for item $i$ (from item tables)
- $usage_{i,r}$: amount of resource $r$ used per unit of item $i$ (from item usage tables, 0 if missing)
- $cap_r$: total available amount of resource $r$ (sum of capacity_ledger entries for $r$)
- $catmin_c$, $catmax_c$, $catfee_c$: minimum_quantity, maximum_quantity, activation_fee_cents for category $c$ (from category constraints table)
- $bonus_{b}$: bonus_cents for bundle $b = (i,j)$ (from bundle bonuses table)
- $incmp_{q}$: incompatible pair $q = (i,j)$ (from incompatible pairs table)
- $req_{p}$: requires pair $p = (i,j)$ (from requires pairs table; $i$ requires $j$)

**Decision Variables**
- $x_i \in \mathbb{Z}_+$: integer quantity of item $i$ to select ($x_i = 0$ if $auth_i = 0$)
- $y_i \in \{0,1\}$: 1 if $x_i > 0$, 0 otherwise (item activation)
- $z_c \in \{0,1\}$: 1 if any $x_i > 0$ for $i$ in category $c$ (category activation)
- $w_b \in \{0,1\}$: 1 if both items in bundle $b$ are selected ($x_{i_b} > 0$ and $x_{j_b} > 0$), 0 otherwise

**Objective**
Maximize net benefit in USD cents:
\[
\max \left(
\sum_{i \in I} unit\_benefit_i \cdot x_i
- \sum_{i \in I} item\_fee_i \cdot y_i
- \sum_{c \in C} catfee_c \cdot z_c
+ \sum_{b \in B} bonus_b \cdot w_b
\right)
\]

**Constraints**

1. **Authorization and Integer Bounds**
   - $x_i = 0$ if $auth_i = 0$ (forbidden options)
   - $x_i \in \{0\} \cup [minlot_i, maxorder_i]$ for $auth_i = 1$ (integer)
   - $y_i = 1$ if $x_i > 0$, $y_i = 0$ if $x_i = 0$ (for all $i$)
   - $y_i \in \{0,1\}$

2. **Resource Capacity (per resource, per area)**
   - For each $r \in R$:
     \[
     \sum_{i \in I: loc_i = r} usage_{i,r} \cdot x_i \leq cap_r
     \]

3. **Category Quantity Limits**
   - For each $c \in C$:
     \[
     catmin_c \leq \sum_{i \in I: cat_i = c} x_i \leq catmax_c
     \]
   - $z_c = 1$ if $\sum_{i \in I: cat_i = c} x_i > 0$, $z_c = 0$ otherwise

4. **Incompatibility**
   - For each $(i,j) \in Q$:
     \[
     y_i + y_j \leq 1
     \]

5. **Requires**
   - For each $(i,j) \in P$:
     \[
     y_i \leq y_j
     \]

6. **Bundle Bonuses**
   - For each $b = (i,j) \in B$:
     \[
     w_b \leq y_i,\quad w_b \leq y_j,\quad w_b \geq y_i + y_j - 1
     \]
     (so $w_b = 1$ iff both $y_i = y_j = 1$)

7. **Linking $y_i$ and $x_i$**
   - For all $i \in I$:
     \[
     x_i \leq maxorder_i \cdot y_i
     \]
     \[
     x_i \geq minlot_i \cdot y_i
     \]
     (for $auth_i = 1$)

8. **Linking $z_c$ and $x_i$**
   - For all $c \in C$:
     \[
     \sum_{i \in I: cat_i = c} x_i \leq catmax_c \cdot z_c
     \]

**Variable Domains**
- $x_i \in \mathbb{Z}_+,\, x_i = 0$ if $auth_i = 0$, else $x_i \in \{0\} \cup [minlot_i, maxorder_i]$
- $y_i \in \{0,1\}$
- $z_c \in \{0,1\}$
- $w_b \in \{0,1\}$

---

## Data Mapping

- **file_5_view_0** and **file_6_view_0**: item options, with columns item_ref, category, authorized, minimum_lot, maximum_order, location_id, unit_benefit_cents, item_fee_cents
- **file_2_view_0**: category constraints, with columns category, minimum_quantity, maximum_quantity, activation_fee_cents
- **file_1_view_0**: resource capacity, with columns resource, entry, amount, unit (sum by resource)
- **file_9_view_0** and **file_10_view_0**: item usage, with columns item_ref, resource, amount, unit
- **file_4_view_0**: incompatible pairs, with columns item_a, item_b
- **file_8_view_0**: requires pairs, with columns item_ref, prerequisite_ref
- **file_0_view_0**: bundle bonuses, with columns item_a, item_b, bonus_cents

All sets, parameters, and constraints are defined directly from these tables as described above.