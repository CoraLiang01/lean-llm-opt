## Mathematical Model

**Sets and Indices**
- $I$: set of item options (all item_ref in item options tables, authorized = 1)
- $G$: set of categories (from category limits table)
- $R$: set of resources (from resource capacity table)
- $B$: set of bundle pairs (from bundle bonuses table)
- $IC$: set of incompatible pairs (from incompatible pairs table)
- $RQ$: set of requires pairs (from requires pairs table)

**Parameters**
- $b_i$: per-unit benefit of item $i$ (sum of amount_cents for each $i$ in benefit components table)
- $f_i$: item activation fee for $i$ (from item fees table)
- $g(i)$: category of item $i$ (from item options tables)
- $a_{ig}$: 1 if $i$ belongs to category $g$, 0 otherwise
- $l_i$, $u_i$: minimum_lot and maximum_order for $i$ (from item options tables)
- $A_i$: 1 if $i$ is authorized, 0 otherwise (from item options tables)
- $c_{ir}$: usage of resource $r$ per unit of $i$ (sum over usage tables, matching item_ref and resource)
- $C_r$: total available amount of resource $r$ (sum of capacity_ledger entries for $r$)
- $L_g$, $U_g$: minimum and maximum quantity for category $g$ (from category limits table)
- $F_g$: category activation fee for $g$ (from category limits table)
- $F_{ij}$: bundle bonus for pair $(i,j)$ (from bundle bonuses table)
- $IC$: set of unordered incompatible pairs $(i,j)$ (from incompatible pairs table)
- $RQ$: set of requires pairs $(i,pr)$ (from requires pairs table; $i$ requires $pr$)

**Decision Variables**
- $x_i \in \mathbb{Z}_+$: quantity of item $i$ selected
- $y_i \in \{0,1\}$: 1 if $x_i > 0$, 0 otherwise (item activation)
- $z_g \in \{0,1\}$: 1 if any $x_i > 0$ for $i$ in $g$, 0 otherwise (category activation)
- $w_{ij} \in \{0,1\}$: 1 if both $x_i > 0$ and $x_j > 0$, 0 otherwise (bundle activation)

**Objective**
Maximize net benefit in USD cents:
\[
\max \left\{
\sum_{i \in I} b_i x_i
- \sum_{i \in I} f_i y_i
- \sum_{g \in G} F_g z_g
+ \sum_{(i,j) \in B} F_{ij} w_{ij}
\right\}
\]

**Constraints**

1. **Authorization and bounds**
   \[
   x_i = 0 \quad \forall i \text{ with } A_i = 0
   \]
   \[
   l_i y_i \leq x_i \leq u_i y_i \quad \forall i \in I
   \]
   \[
   y_i \in \{0,1\},\ x_i \in \mathbb{Z}_+ \quad \forall i \in I
   \]

2. **Resource capacities**
   \[
   \sum_{i \in I} c_{ir} x_i \leq C_r \quad \forall r \in R
   \]

3. **Category quantity limits and activation**
   \[
   L_g z_g \leq \sum_{i \in I: g(i)=g} x_i \leq U_g z_g \quad \forall g \in G
   \]
   \[
   z_g \in \{0,1\} \quad \forall g \in G
   \]

4. **Incompatibility**
   \[
   y_i + y_j \leq 1 \quad \forall (i,j) \in IC
   \]

5. **Requires**
   \[
   y_i \leq y_{pr} \quad \forall (i,pr) \in RQ
   \]

6. **Bundle bonuses**
   \[
   w_{ij} \leq y_i,\quad w_{ij} \leq y_j,\quad w_{ij} \geq y_i + y_j - 1 \quad \forall (i,j) \in B
   \]
   \[
   w_{ij} \in \{0,1\} \quad \forall (i,j) \in B
   \]

**Data Mapping**

- file_0_view_0: benefit components ($b_i$)
- file_1_view_0: bundle bonuses ($F_{ij}$, $B$)
- file_2_view_0: resource capacity ledger ($C_r$)
- file_3_view_0: category limits ($L_g$, $U_g$, $F_g$)
- file_4_view_0: item identity (for display only)
- file_5_view_0: incompatible pairs ($IC$)
- file_6_view_0, file_7_view_0: item options ($I$, $A_i$, $l_i$, $u_i$, $g(i)$)
- file_8_view_0: item fees ($f_i$)
- file_10_view_0: requires pairs ($RQ$)
- file_11_view_0, file_12_view_0: usage ($c_{ir}$)

**Variable domains and all constraints are as above.**