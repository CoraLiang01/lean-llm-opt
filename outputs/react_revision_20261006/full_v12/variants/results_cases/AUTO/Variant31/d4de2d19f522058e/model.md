## Symbolic Mathematical Model

**Sets**
- $I$: set of item offers (from file_5_view_0, column item_ref, where authorized = 1)
- $C$: set of categories (from file_2_view_0, column category)
- $R$: set of resources (from file_1_view_0, column resource)
- $B$: set of bundle bonus pairs (from file_0_view_0, columns item_a, item_b)
- $P$: set of incompatible item pairs (from file_4_view_0, columns item_a, item_b)
- $Q$: set of requires dependencies (from file_7_view_0, columns item_ref, prerequisite_ref)

**Parameters**
- $u_i$: unit_benefit_cents for item $i$ (file_5_view_0)
- $f_i$: item_fee_cents for item $i$ (file_5_view_0)
- $cat(i)$: category of item $i$ (file_5_view_0)
- $min_i$, $max_i$: minimum_lot, maximum_order for item $i$ (file_5_view_0)
- $a_c$, $b_c$: minimum_quantity, maximum_quantity for category $c$ (file_2_view_0)
- $F_c$: activation_fee_cents for category $c$ (file_2_view_0)
- $U_{ir}$: usage of resource $r$ per unit of item $i$ (file_8_view_0, default 0 if missing)
- $L_r$: total available amount of resource $r$ (sum of file_1_view_0, entry = "opening" + all "reservation" for $r$)
- $S_{ij}$: 1 if $(i,j)\in P$ or $(j,i)\in P$, 0 otherwise (incompatibility)
- $D_{ij}$: 1 if $(i,j)\in Q$, 0 otherwise (requires)
- $bb_{ij}$: bundle bonus_cents for $(i,j)\in B$, 0 otherwise
- $I_{cat}$: set of items in category $cat$ (from file_5_view_0)
- $M$: a sufficiently large constant (e.g., $M = \max_{i} max_i$)

**Decision Variables**
- $x_i \in \mathbb{Z}_+$: quantity of item $i$ to order
- $y_i \in \{0,1\}$: 1 if $x_i > 0$, 0 otherwise (item activation)
- $z_c \in \{0,1\}$: 1 if any $x_i > 0$ for $i \in I_{cat}$ (category activation)
- $w_{ij} \in \{0,1\}$: 1 if both $x_i > 0$ and $x_j > 0$ (bundle bonus activation), for $(i,j)\in B$

**Objective**
Maximize net benefit in USD cents:
\[
\max \left\{
\sum_{i \in I} u_i x_i
- \sum_{i \in I} f_i y_i
- \sum_{c \in C} F_c z_c
+ \sum_{(i,j) \in B} bb_{ij} w_{ij}
\right\}
\]

**Constraints**

1. **Item authorization and lot/order bounds**
   - $x_i = 0$ if item $i$ is unauthorized (from file_5_view_0, authorized = 0)
   - $min_i y_i \leq x_i \leq max_i y_i$ for all $i \in I$
   - $y_i \in \{0,1\}$, $x_i \in \mathbb{Z}_+$

2. **Category quantity limits (unconditional)**
   - $a_c \leq \sum_{i \in I_{cat}} x_i \leq b_c$ for all $c \in C$

3. **Category activation**
   - $z_c \geq y_i$ for all $i \in I_{cat}$, $c = cat(i)$
   - $z_c \in \{0,1\}$

4. **Resource capacity**
   - $\sum_{i \in I} U_{ir} x_i \leq L_r$ for all $r \in R$

5. **Incompatibility**
   - $y_i + y_j \leq 1$ for all $(i,j) \in P$

6. **Requires dependencies**
   - $x_i \leq M y_{prereq}$ for all $(i,prereq) \in Q$
   - $y_i \leq y_{prereq}$ for all $(i,prereq) \in Q$

7. **Bundle bonuses**
   - $w_{ij} \leq y_i$, $w_{ij} \leq y_j$, $w_{ij} \geq y_i + y_j - 1$ for all $(i,j) \in B$
   - $w_{ij} \in \{0,1\}$

**Data Mapping**

- file_0_view_0: bundle bonuses $B$, $bb_{ij}$
- file_1_view_0: resource ledger $L_r$
- file_2_view_0: category limits $a_c$, $b_c$, activation fees $F_c$
- file_3_view_0: item identity (for display only)
- file_4_view_0: incompatible pairs $P$
- file_5_view_0: item offers $I$, $u_i$, $f_i$, $cat(i)$, $min_i$, $max_i$, authorization
- file_7_view_0: requires dependencies $Q$
- file_8_view_0: resource usage $U_{ir}$

**Notes**
- All sums, sets, and indices are over the current rows as described above.
- All units are in USD cents and matching base units as per the data.
- All fixed fees and bonuses are charged once per activation as described.
- All constraints are enforced for the current plan only.