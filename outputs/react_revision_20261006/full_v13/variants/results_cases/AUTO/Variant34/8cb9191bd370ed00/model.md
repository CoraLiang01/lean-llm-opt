## Symbolic Mathematical Model

**Sets and Indices**
- $I$: set of item options (item_ref), as listed in item options tables (authorized or not)
- $I_{auth} \subseteq I$: set of authorized item options (authorized = 1)
- $R$: set of resources (e.g., SECTION_A, SECTION_B, SECTION_C)
- $C$: set of categories (from category limits table)
- $B$: set of bundle pairs $(i,j)$ (from bundle bonuses table)
- $Q$: set of incompatible pairs $(i,j)$ (from incompatible pairs table)
- $P$: set of requires pairs $(i,pr)$ (from requires pairs table)

**Parameters**
- $b_i$: per-pack net benefit for item $i$ (sum of amount_cents for all components with item_ref $i$) [file_0_view_0]
- $f_i$: fixed item fee for item $i$ (activation_fee_cents) [file_8_view_0]
- $cat_i$: category of item $i$ [file_6_view_0, file_7_view_0]
- $loc_i$: location/section of item $i$ [file_6_view_0, file_7_view_0]
- $min_i$, $max_i$: minimum lot and maximum order for item $i$ [file_6_view_0, file_7_view_0]
- $a_i$: 1 if item $i$ is authorized, 0 otherwise [file_6_view_0, file_7_view_0]
- $u_{ir}$: per-pack usage of resource $r$ by item $i$ in ml (convert from liters if needed) [file_11_view_0, file_12_view_0]
- $cap_r$: available capacity for resource $r$ in ml (sum of capacity_ledger entries for $r$) [file_2_view_0]
- $catmin_c$, $catmax_c$: minimum and maximum total packs for category $c$ [file_3_view_0]
- $catfee_c$: activation fee for category $c$ [file_3_view_0]
- $f_{ij}^{bundle}$: bundle bonus for pair $(i,j)$ [file_1_view_0]
- $f_{ij}^{inc}$: 1 if $(i,j)$ is an incompatible pair, 0 otherwise [file_5_view_0]
- $pr_i$: set of prerequisite items for $i$ (from requires pairs) [file_10_view_0]

**Decision Variables**
- $x_i \in \mathbb{Z}_{\geq 0}$: number of packs of item $i$ to display
- $y_i \in \{0,1\}$: 1 if $x_i > 0$, 0 otherwise (item activation)
- $z_c \in \{0,1\}$: 1 if any item in category $c$ is selected, 0 otherwise (category activation)
- $w_{ij} \in \{0,1\}$: 1 if both $x_i > 0$ and $x_j > 0$ for bundle $(i,j)$, 0 otherwise

**Objective**
Maximize net merchandising benefit (in USD cents):
\[
\max \Bigg\{
\sum_{i \in I_{auth}} b_i x_i
- \sum_{i \in I_{auth}} f_i y_i
- \sum_{c \in C} catfee_c z_c
+ \sum_{(i,j) \in B} f_{ij}^{bundle} w_{ij}
\Bigg\}
\]

**Constraints**

1. **Authorization and Integer Bounds**
   \[
   x_i = 0 \quad \forall i \in I \setminus I_{auth}
   \]
   \[
   x_i \in \{0\} \cup [min_i, max_i] \cap \mathbb{Z} \quad \forall i \in I_{auth}
   \]

2. **Item Activation**
   \[
   y_i \in \{0,1\} \quad \forall i \in I_{auth}
   \]
   \[
   x_i \leq max_i y_i \quad \forall i \in I_{auth}
   \]
   \[
   x_i \geq min_i y_i \quad \forall i \in I_{auth}
   \]

3. **Section Resource Capacity**
   \[
   \sum_{i \in I_{auth}} u_{ir} x_i \leq cap_r \quad \forall r \in R
   \]

4. **Category Quantity Limits and Activation**
   \[
   catmin_c z_c \leq \sum_{i \in I_{auth}: cat_i = c} x_i \leq catmax_c z_c \quad \forall c \in C
   \]
   \[
   z_c \in \{0,1\} \quad \forall c \in C
   \]
   \[
   y_i \leq z_{cat_i} \quad \forall i \in I_{auth}
   \]

5. **Incompatible Pairs**
   \[
   y_i + y_j \leq 1 \quad \forall (i,j) \in Q
   \]

6. **Requires Pairs**
   \[
   y_i \leq y_{pr} \quad \forall (i,pr) \in P
   \]

7. **Bundle Bonuses**
   \[
   w_{ij} \leq y_i, \quad w_{ij} \leq y_j, \quad w_{ij} \geq y_i + y_j - 1 \quad \forall (i,j) \in B
   \]
   \[
   w_{ij} = 0 \quad \text{if $i$ or $j$ is not authorized}
   \]

**Data Mapping**

- $I$, $I_{auth}$: All item_ref from [file_6_view_0] and [file_7_view_0], with authorized = 1 for $I_{auth}$
- $b_i$: sum of amount_cents for each item_ref $i$ in [file_0_view_0]
- $f_i$: activation_fee_cents for item_ref $i$ in [file_8_view_0]
- $cat_i$, $loc_i$, $min_i$, $max_i$, $a_i$: from [file_6_view_0] and [file_7_view_0]
- $u_{ir}$: sum of amount for item_ref $i$ and resource $r$ in [file_11_view_0] and [file_12_view_0], converted to ml if unit is liter (multiply by 1000)
- $cap_r$: sum of amount for resource $r$ in [file_2_view_0]
- $catmin_c$, $catmax_c$, $catfee_c$: from [file_3_view_0]
- $f_{ij}^{bundle}$: bonus_cents for each (item_a, item_b) in [file_1_view_0]
- $Q$: all (item_a, item_b) in [file_5_view_0]
- $P$: all (item_ref, prerequisite_ref) in [file_10_view_0]

**Variable Domains**
- $x_i \in \{0\} \cup [min_i, max_i] \cap \mathbb{Z}$ for $i \in I_{auth}$; $x_i = 0$ for $i \notin I_{auth}$
- $y_i, z_c, w_{ij} \in \{0,1\}$ as above

**Units**
- All benefits, fees, and bonuses in USD cents
- All resource usage and capacities in ml

**Notes**
- All summations and constraints are over the current, supplied item options and data rows only.
- All fixed and bundle bonuses are counted once per activation/selection as described.
- All constraints and parameters are mapped directly from the supplied tables as described above.