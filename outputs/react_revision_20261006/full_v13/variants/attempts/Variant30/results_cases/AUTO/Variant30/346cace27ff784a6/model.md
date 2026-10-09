## Symbolic Mathematical Model

Let:

- $I$ = set of items (options) available for replenishment, indexed by $i$
- $C$ = set of categories, indexed by $c$
- $R$ = set of resources, indexed by $r$
- $B$ = set of bundle pairs, indexed by $(i,j)$
- $P$ = set of incompatible pairs, indexed by $(i,j)$
- $Q$ = set of requires pairs, indexed by $(i,j)$, where $i$ requires $j$

**Parameters (all in USD cents):**
- $b_i$: per-unit net benefit for item $i$ (sum of all signed amount_cents for $i$)
- $f_i$: item_fee for item $i$ (deducted once if $x_i > 0$)
- $a_i$: 1 if item $i$ is authorized, 0 otherwise
- $l_i$: minimum_lot for item $i$
- $u_i$: maximum_order for item $i$
- $cat(i)$: category of item $i$
- $L_c$: minimum_quantity for category $c$
- $U_c$: maximum_quantity for category $c$
- $F_c$: activation_fee_cents for category $c$
- $q_{i,r}$: usage of resource $r$ per unit of item $i$ (in resource's native units)
- $K_r$: total available capacity for resource $r$ (in resource's native units)
- $S_r$: scaling factor to convert $q_{i,r}$ and $K_r$ to the same units (e.g., 1000 for liters to ml, 60 for hours to minutes, 1000 for kwh to wh)
- $F$: set of all options $i$ with $a_i = 1$
- $B_{ij}$: bundle bonus_cents for bundle $(i,j)$
- $F_{cat}$: set of all categories used by at least one $i$ with $x_i > 0$

**Decision Variables:**
- $x_i \in \mathbb{Z}_+$: integer replenishment quantity for item $i$
- $y_i \in \{0,1\}$: 1 if $x_i > 0$, 0 otherwise (item activation indicator)
- $z_c \in \{0,1\}$: 1 if any $x_i > 0$ for $i$ with $cat(i) = c$ (category activation indicator)
- $w_{ij} \in \{0,1\}$: 1 if both $x_i > 0$ and $x_j > 0$ (bundle activation indicator)

**Objective:**
\[
\max \left\{
\sum_{i \in I} b_i x_i
- \sum_{i \in I} f_i y_i
- \sum_{c \in C} F_c z_c
+ \sum_{(i,j) \in B} B_{ij} w_{ij}
\right\}
\]

**Subject to:**

1. **Authorization and Lot Constraints:**
   \[
   l_i y_i \leq x_i \leq u_i y_i \quad \forall i \in I
   \]
   \[
   x_i = 0 \quad \text{if } a_i = 0
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]
   \[
   x_i \in \mathbb{Z}_+ \quad \forall i \in I
   \]

2. **Category Quantity Limits:**
   \[
   L_c z_c \leq \sum_{i:cat(i)=c} x_i \leq U_c z_c \quad \forall c \in C
   \]
   \[
   z_c \geq y_i \quad \forall i \in I,\, cat(i)=c
   \]
   \[
   z_c \in \{0,1\} \quad \forall c \in C
   \]

3. **Resource Capacity Constraints:**
   \[
   \sum_{i \in I} q_{i,r} S_r x_i \leq K_r \quad \forall r \in R
   \]
   (Convert all $q_{i,r}$ and $K_r$ to the same units using $S_r$.)

4. **Incompatibility Constraints:**
   \[
   y_i + y_j \leq 1 \quad \forall (i,j) \in P
   \]

5. **Requires Constraints:**
   \[
   y_i \leq y_j \quad \forall (i,j) \in Q
   \]
   (i.e., if $x_i > 0$ then $x_j > 0$.)

6. **Bundle Bonus Activation:**
   \[
   w_{ij} \leq y_i
   \]
   \[
   w_{ij} \leq y_j
   \]
   \[
   w_{ij} \geq y_i + y_j - 1
   \]
   \[
   w_{ij} \in \{0,1\} \quad \forall (i,j) \in B
   \]

**Data Mapping:**

- $I$: All items from the latest (highest revision, non-DELETE) records in the item tables (file_9_view_0 and file_12_view_0), after applying the selection rule.
- $C$: All categories from the latest (highest revision, non-DELETE) records in the category tables (file_15_view_0 and file_17_view_0).
- $R$: All resources from the usage and capacity_ledger tables (file_5_view_0, file_7_view_0, file_18_view_0, file_19_view_0, file_22_view_0).
- $B$: All bundle pairs from the latest (highest revision, non-DELETE) records in the bundle tables (file_3_view_0 and file_10_view_0).
- $P$: All incompatible pairs from the latest (highest revision, non-DELETE) records in the incompatible tables (file_6_view_0 and file_13_view_0).
- $Q$: All requires pairs from the latest (highest revision, non-DELETE) records in the requires tables (file_0_view_0 and file_23_view_0).
- $b_i$: For each $i$, sum all amount_cents from benefit tables (file_4_view_0, file_11_view_0, file_14_view_0) for that $i$.
- $f_i$: For each $i$, activation_fee_cents from item_fee tables (file_2_view_0 and file_16_view_0).
- $a_i$, $l_i$, $u_i$, $cat(i)$: From item tables (file_9_view_0 and file_12_view_0).
- $L_c$, $U_c$, $F_c$: From category tables (file_15_view_0 and file_17_view_0).
- $q_{i,r}$: From usage tables (file_5_view_0, file_18_view_0, file_22_view_0), latest revision per (item_ref, resource).
- $K_r$: Sum of all amount for resource $r$ in capacity_ledger tables (file_7_view_0 and file_19_view_0), using only the latest revision per (resource, entry).
- $S_r$: Scaling factors: 1000 for liters to ml, 60 for hours to minutes, 1000 for kwh to wh.
- $B_{ij}$: bonus_cents from bundle tables (file_3_view_0 and file_10_view_0).
- $F_{cat}$: All categories $c$ with $z_c = 1$.

**Maximize the net benefit in USD cents, subject to all above constraints and data mappings.**