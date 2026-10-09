## Symbolic Mathematical Model

**Sets:**
- $I$: set of items (from identity tables, after filtering and mapping)
- $C$: set of categories (from category tables)
- $R$: set of resources (from usage/capacity_ledger tables)
- $P$: set of item pairs with bundle bonuses
- $Q$: set of item pairs with incompatibility constraints
- $D$: set of (item, prerequisite) pairs with requires dependencies

**Parameters:**
- $b_i$: per-unit net benefit (sum of amount_cents from all benefit rows for item $i$)
- $f_i$: item fee (from item_fee table, activation_fee_cents for item $i$)
- $a_i$: 1 if item $i$ is authorized, 0 otherwise (from item tables)
- $l_i$, $u_i$: minimum_lot, maximum_order for item $i$ (from item tables)
- $g_i$: category of item $i$ (from item tables)
- $L_c$, $U_c$: minimum_quantity, maximum_quantity for category $c$ (from category tables)
- $F_c$: activation_fee_cents for category $c$ (from category tables)
- $s_{ir}$: usage of resource $r$ per unit of item $i$ (from usage tables, with unit conversion)
- $K_r$: total available capacity for resource $r$ (sum of capacity_ledger entries for $r$, with unit conversion)
- $B_{ij}$: bundle bonus_cents for pair $(i,j)$ (from bundle tables)
- $Q$: set of unordered incompatible item pairs $(i,j)$ (from incompatible tables)
- $D$: set of requires pairs $(i,j)$ (from requires tables; $i$ requires $j$)

**Decision Variables:**
- $x_i \in \mathbb{Z}_+$: integer quantity ordered for item $i$
- $y_i \in \{0,1\}$: 1 if $x_i > 0$, 0 otherwise (item activation indicator)
- $z_c \in \{0,1\}$: 1 if any $x_i > 0$ for $i$ in category $c$, 0 otherwise (category activation indicator)
- $w_{ij} \in \{0,1\}$: 1 if both $x_i > 0$ and $x_j > 0$, 0 otherwise (bundle activation indicator)

**Objective:**
Maximize net benefit in USD cents:
\[
\max \left\{
\sum_{i \in I} b_i x_i
- \sum_{i \in I} f_i y_i
- \sum_{c \in C} F_c z_c
+ \sum_{(i,j) \in P} B_{ij} w_{ij}
\right\}
\]

**Constraints:**

1. **Authorization and Lot Size:**
   \[
   l_i y_i \leq x_i \leq u_i y_i \quad \forall i \in I
   \]
   \[
   x_i = 0 \quad \text{if } a_i = 0
   \]
   \[
   y_i \in \{0,1\},\ x_i \in \mathbb{Z}_+ \quad \forall i \in I
   \]

2. **Category Quantity Limits:**
   \[
   L_c z_c \leq \sum_{i: g_i = c} x_i \leq U_c z_c \quad \forall c \in C
   \]
   \[
   z_c \geq y_i \quad \forall i \in I,\ g_i = c
   \]

3. **Resource Capacity:**
   \[
   \sum_{i \in I} s_{ir} x_i \leq K_r \quad \forall r \in R
   \]
   (Convert all $s_{ir}$ and $K_r$ to common units: 1000 ml = 1 liter, 60 min = 1 hour, 1000 wh = 1 kwh.)

4. **Incompatibility:**
   \[
   y_i + y_j \leq 1 \quad \forall (i,j) \in Q
   \]

5. **Requires Dependencies:**
   \[
   x_i \leq M_{ij} y_j \quad \forall (i,j) \in D
   \]
   (Or, equivalently: $y_i \leq y_j$ for all $(i,j) \in D$; $M_{ij}$ is a large enough upper bound.)

6. **Bundle Bonuses:**
   \[
   w_{ij} \leq y_i,\quad w_{ij} \leq y_j,\quad w_{ij} \geq y_i + y_j - 1 \quad \forall (i,j) \in P
   \]

**Data Mapping:**
- $I$: All items from identity tables after filtering and mapping, joined with item, benefit, item_fee, usage, etc.
- $C$: All categories from category tables.
- $R$: All resources from usage/capacity_ledger tables.
- $b_i$: Sum of amount_cents from benefit tables for each item_ref $i$ (files 4, 11, 14).
- $f_i$: activation_fee_cents from item_fee tables for each item_ref $i$ (files 2, 16).
- $a_i$, $l_i$, $u_i$, $g_i$: from item tables (files 9, 12).
- $L_c$, $U_c$, $F_c$: from category tables (files 15, 17).
- $s_{ir}$: from usage tables (files 5, 18, 22), with unit conversion.
- $K_r$: sum of amount from capacity_ledger tables (files 7, 19), with unit conversion.
- $B_{ij}$: from bundle tables (files 3, 10).
- $Q$: from incompatible tables (files 6, 13).
- $D$: from requires tables (files 0, 23).

**Notes:**
- All summation and index sets are defined by the current filtered and mapped data.
- All units are harmonized as per the question (ml/liter, min/hour, wh/kwh).
- All fixed fees and bonuses are in USD cents.
- All constraints and variable domains are enforced as described.
- The maximum net benefit is reported in USD cents.