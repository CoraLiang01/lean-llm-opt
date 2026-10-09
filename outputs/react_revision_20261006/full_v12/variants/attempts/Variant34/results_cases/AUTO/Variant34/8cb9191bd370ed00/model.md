## Symbolic Mathematical Model

**Sets:**
- $I$: set of item options (item_ref, location_id) with authorized $=1$ (from tables 6 and 7)
- $R$: set of resources (from table 2)
- $C$: set of categories (from table 3)
- $B$: set of bundle pairs (from table 1)
- $P$: set of incompatible pairs (from table 5)
- $Q$: set of requires pairs (from table 10)

**Parameters:**
- $b_i$: per-unit net benefit for item $i$ (sum of amount_cents for $i$ in table 0)
- $f_i$: item_fee for item $i$ (from table 8; 0 if missing)
- $cat(i)$: category of item $i$ (from tables 6/7)
- $loc(i)$: location_id of item $i$ (from tables 6/7)
- $min_i$, $max_i$: minimum_lot, maximum_order for item $i$ (from tables 6/7)
- $auth_i$: authorized flag for item $i$ (from tables 6/7)
- $a_{ir}$: per-pack usage of resource $r$ by item $i$ in ml (sum over tables 11/12, convert liters to ml)
- $L_r$: available capacity for resource $r$ (sum of amount in table 2 for each $r$)
- $catmin_c$, $catmax_c$, $catfee_c$: minimum_quantity, maximum_quantity, activation_fee_cents for category $c$ (from table 3)
- $F_{ij}$: bundle bonus_cents for bundle $(i,j)$ (from table 1)
- $A_{ij}$: 1 if $(i,j)$ is an incompatible pair (from table 5), else 0
- $Q_{ik}$: 1 if $(i,k)$ is a requires pair (from table 10), else 0

**Decision Variables:**
- $x_i \in \mathbb{Z}_+$: number of packs of item $i$ to display
- $y_i \in \{0,1\}$: 1 if $x_i > 0$, 0 otherwise (item activation)
- $z_c \in \{0,1\}$: 1 if any $x_i > 0$ for $i$ in category $c$ (category activation)
- $w_{ij} \in \{0,1\}$: 1 if both $x_i > 0$ and $x_j > 0$ for bundle $(i,j)$, 0 otherwise

**Objective:**
\[
\max \left\{
\sum_{i \in I} b_i x_i
- \sum_{i \in I} f_i y_i
- \sum_{c \in C} catfee_c \cdot z_c
+ \sum_{(i,j) \in B} F_{ij} w_{ij}
\right\}
\]

**Constraints:**

1. **Authorization and bounds:**
   - $x_i = 0$ if $auth_i = 0$
   - $x_i = 0$ or $min_i \leq x_i \leq max_i$ if $auth_i = 1$
   - $x_i \in \mathbb{Z}_+$

2. **Section (resource) capacity:**
   - For each $r \in R$:
     \[
     \sum_{i \in I} a_{ir} x_i \leq L_r
     \]

3. **Category quantity limits:**
   - For each $c \in C$:
     \[
     catmin_c \leq \sum_{i \in I: cat(i)=c} x_i \leq catmax_c
     \]

4. **Category activation:**
   - For each $c \in C$:
     \[
     z_c \geq y_i \quad \forall i \in I: cat(i)=c
     \]
     \[
     y_i \geq \frac{x_i}{max_i} \quad \forall i \in I
     \]
     \[
     y_i \leq \text{sign}(x_i) \leq 1
     \]
     (Or, equivalently, $y_i = 1$ if $x_i > 0$, $y_i = 0$ if $x_i = 0$.)

5. **Bundles:**
   - For each $(i,j) \in B$:
     \[
     w_{ij} \leq y_i
     \]
     \[
     w_{ij} \leq y_j
     \]
     \[
     w_{ij} \geq y_i + y_j - 1
     \]

6. **Incompatibility:**
   - For each $(i,j) \in P$:
     \[
     y_i + y_j \leq 1
     \]

7. **Requires:**
   - For each $(i,k) \in Q$:
     \[
     y_i \leq y_k
     \]

8. **Variable domains:**
   - $x_i \in \{0\} \cup [min_i, max_i] \cap \mathbb{Z}$ if $auth_i = 1$; $x_i = 0$ if $auth_i = 0$
   - $y_i, z_c, w_{ij} \in \{0,1\}$

**Data Mapping:**
- Table file_0_view_0: $b_i$ (sum of amount_cents per item_ref)
- Table file_8_view_0: $f_i$ (activation_fee_cents per item_ref)
- Table file_6_view_0 and file_7_view_0: $I$, $cat(i)$, $loc(i)$, $min_i$, $max_i$, $auth_i$
- Table file_2_view_0: $L_r$ (sum of amount per resource)
- Table file_11_view_0 and file_12_view_0: $a_{ir}$ (sum per item_ref, resource; convert liters to ml)
- Table file_3_view_0: $catmin_c$, $catmax_c$, $catfee_c$
- Table file_1_view_0: $F_{ij}$ (bundle bonuses)
- Table file_5_view_0: $A_{ij}$ (incompatible pairs)
- Table file_10_view_0: $Q_{ik}$ (requires pairs)

**Notes:**
- All sums, indices, and parameters are over the current authorized options and data as described.
- All monetary values are in USD cents.
- All resource usage is in ml.
- All constraints and variables are as above; unauthorized options are fixed at zero.