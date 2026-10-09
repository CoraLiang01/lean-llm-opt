## Symbolic Mathematical Model

**Sets**
- $I$: set of item options (item_ref from item tables, union of all authorized items)
- $G$: set of categories (category table)
- $R$: set of resources (resource in usage/capacity_ledger tables)
- $B$: set of bundles (rows in bundle table)
- $C$: set of incompatible pairs (rows in incompatible table)
- $Q$: set of requires pairs (rows in requires table)

**Parameters**
- $b_i$: total per-unit benefit (sum of amount_cents for each item_ref $i$ in benefit table)
- $f_i$: item activation fee (activation_fee_cents for $i$ in item_fee table)
- $g(i)$: category of item $i$ (from item tables)
- $a_g$: category activation fee (activation_fee_cents for $g$ in category table)
- $l_g, u_g$: category min/max quantity (minimum_quantity, maximum_quantity for $g$)
- $m_i, M_i$: item min/max order (minimum_lot, maximum_order for $i$)
- $A_i$: 1 if item $i$ is authorized, 0 otherwise (from item tables)
- $u_{ir}$: per-unit usage of resource $r$ by item $i$ (from usage tables, 0 if missing)
- $S_r$: available resource $r$ (sum of amount for resource $r$ in capacity_ledger table)
- $(i,j), \beta_{ij}$: bundle pairs and bonus_cents (from bundle table)
- $(i,j) \in C$: incompatible pairs (from incompatible table)
- $(i,j) \in Q$: requires pairs (from requires table; $i$ requires $j$)

**Decision Variables**
- $x_i \in \mathbb{Z}_+$: quantity of item $i$ selected
- $y_i \in \{0,1\}$: 1 if $x_i > 0$, 0 otherwise
- $z_g \in \{0,1\}$: 1 if any item in category $g$ is selected, 0 otherwise
- $w_{ij} \in \{0,1\}$: 1 if both $x_i > 0$ and $x_j > 0$, 0 otherwise (for each bundle $(i,j)$)

**Objective**
\[
\max \left\{
\sum_{i \in I} b_i x_i
- \sum_{i \in I} f_i y_i
- \sum_{g \in G} a_g z_g
+ \sum_{(i,j) \in B} \beta_{ij} w_{ij}
\right\}
\]

**Constraints**

1. **Authorization and Order Bounds**
   \[
   m_i y_i \leq x_i \leq M_i y_i \quad \forall i \in I
   \]
   \[
   x_i = 0,\, y_i = 0 \quad \text{if } A_i = 0
   \]

2. **Category Quantity Bounds**
   \[
   l_g z_g \leq \sum_{i \in I: g(i) = g} x_i \leq u_g z_g \quad \forall g \in G
   \]
   \[
   z_g \geq y_i \quad \forall i \in I,\, g(i) = g
   \]

3. **Resource Capacity**
   \[
   \sum_{i \in I} u_{ir} x_i \leq S_r \quad \forall r \in R
   \]

4. **Incompatibility**
   \[
   y_i + y_j \leq 1 \quad \forall (i,j) \in C
   \]

5. **Requires**
   \[
   y_i \leq y_j \quad \forall (i,j) \in Q
   \]

6. **Bundle Bonuses**
   \[
   w_{ij} \leq y_i,\quad w_{ij} \leq y_j,\quad w_{ij} \geq y_i + y_j - 1 \quad \forall (i,j) \in B
   \]

7. **Variable Domains**
   \[
   x_i \in \{0\} \cup [m_i, M_i] \cap \mathbb{Z} \quad \forall i \in I
   \]
   \[
   y_i, z_g, w_{ij} \in \{0,1\}
   \]

---

## Data Mapping

- $I$: All item_ref with $A_i = 1$ from item tables (file_6_view_0, file_7_view_0)
- $b_i$: sum of amount_cents for item_ref $i$ in file_0_view_0 (benefit)
- $f_i$: activation_fee_cents for item_ref $i$ in file_8_view_0 (item_fee)
- $g(i)$: category for item_ref $i$ in item tables (file_6_view_0, file_7_view_0)
- $a_g$: activation_fee_cents for category $g$ in file_3_view_0 (category)
- $l_g, u_g$: minimum_quantity, maximum_quantity for $g$ in file_3_view_0 (category)
- $m_i, M_i$: minimum_lot, maximum_order for item_ref $i$ in item tables (file_6_view_0, file_7_view_0)
- $A_i$: authorized for item_ref $i$ in item tables (file_6_view_0, file_7_view_0)
- $u_{ir}$: amount for (item_ref $i$, resource $r$) in file_11_view_0 and file_12_view_0 (usage), 0 if missing
- $S_r$: sum of amount for resource $r$ in file_2_view_0 (capacity_ledger)
- $B$: pairs (item_a, item_b) and bonus_cents in file_1_view_0 (bundle)
- $C$: pairs (item_a, item_b) in file_5_view_0 (incompatible)
- $Q$: pairs (item_ref, prerequisite_ref) in file_10_view_0 (requires)

All indices, parameters, and constraints are mapped directly from the supplied tables as described.