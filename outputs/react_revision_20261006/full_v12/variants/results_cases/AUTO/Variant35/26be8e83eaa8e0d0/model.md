## Symbolic Mathematical Model

**Sets:**
- $I$: set of item options (from file_7_view_0, column item_ref, where authorized = 1)
- $C$: set of categories (from file_3_view_0, column category)
- $L$: set of platforms/resources (from file_2_view_0, column resource)
- $B$: set of bundle pairs (from file_1_view_0, columns item_a, item_b)
- $P$: set of incompatible pairs (from file_6_view_0, columns item_a, item_b)
- $Q$: set of requires pairs (from file_10_view_0, columns item_ref, prerequisite_ref)

**Parameters:**
- $b_i$: net per-unit benefit in USD cents for item $i \in I$ (sum over all components for $i$ in file_0_view_0, converted to USD cents using file_4_view_0)
- $f_i$: item fee in USD cents for $i$ (from file_8_view_0, activation_fee_cents)
- $cat(i)$: category of $i$ (from file_7_view_0, category)
- $loc(i)$: platform/resource of $i$ (from file_7_view_0, location_id)
- $min_i$, $max_i$: minimum lot, maximum order for $i$ (from file_7_view_0, minimum_lot, maximum_order)
- $u_{i\ell}$: memory usage per unit of $i$ on resource $\ell$ in MB (from file_11_view_0, amount × 1000 if unit = GB, for matching item_ref and resource)
- $F_c$: category activation fee in USD cents (from file_3_view_0, activation_fee_cents)
- $min_c$, $max_c$: category lower/upper quantity limits (from file_3_view_0, minimum_quantity, maximum_quantity)
- $cap_\ell$: available memory for resource $\ell$ in MB (sum of opening and reservation entries for $\ell$ in file_2_view_0)
- $bonus_{ab}$: bundle bonus in USD cents for $(a,b) \in B$ (from file_1_view_0, bonus_cents)
- $P$: set of unordered incompatible pairs $(i,j)$ (from file_6_view_0)
- $Q$: set of requires pairs $(i,prereq)$ (from file_10_view_0)

**Decision Variables:**
- $x_i \in \{0\} \cup \{\text{integers}: min_i \leq x_i \leq max_i\}$ for $i \in I$ (quantity of item $i$)
- $y_i \in \{0,1\}$ for $i \in I$ (1 if $x_i > 0$, 0 otherwise)
- $z_c \in \{0,1\}$ for $c \in C$ (1 if any $x_i > 0$ for $i$ with $cat(i)=c$, 0 otherwise)
- $w_{ab} \in \{0,1\}$ for $(a,b) \in B$ (1 if both $x_a > 0$ and $x_b > 0$, 0 otherwise)

**Objective:**
\[
\max \Bigg\{ \sum_{i \in I} \left[ b_i x_i - f_i y_i \right] + \sum_{c \in C} (-F_c z_c) + \sum_{(a,b) \in B} bonus_{ab} w_{ab} \Bigg\}
\]

**Constraints:**

1. **Authorization and bounds:**
   \[
   x_i = 0 \quad \text{if $i$ is unauthorized (from file_7_view_0, authorized = 0)}
   \]
   \[
   x_i = 0 \text{ or } min_i \leq x_i \leq max_i, \quad x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]
   \[
   y_i = 1 \iff x_i > 0, \quad y_i \in \{0,1\}
   \]

2. **Category limits:**
   \[
   min_c \leq \sum_{i \in I: cat(i)=c} x_i \leq max_c \quad \forall c \in C
   \]
   \[
   z_c \geq y_i \quad \forall i \in I: cat(i)=c
   \]
   \[
   z_c \in \{0,1\}
   \]

3. **Resource (memory) constraints:**
   \[
   \sum_{i \in I: loc(i)=\ell} u_{i\ell} x_i \leq cap_\ell \quad \forall \ell \in L
   \]

4. **Incompatibility:**
   \[
   y_i + y_j \leq 1 \quad \forall (i,j) \in P
   \]

5. **Requires:**
   \[
   y_i \leq y_{prereq} \quad \forall (i,prereq) \in Q
   \]

6. **Bundle bonuses:**
   \[
   w_{ab} \leq y_a, \quad w_{ab} \leq y_b, \quad w_{ab} \geq y_a + y_b - 1 \quad \forall (a,b) \in B
   \]

**Data Mapping:**
- $I$: file_7_view_0, item_ref, where authorized = 1
- $C$: file_3_view_0, category
- $L$: file_2_view_0, resource
- $b_i$: sum over file_0_view_0, amount × (usd_cents_numerator/denominator) for each component of $i$, using file_4_view_0 for currency conversion
- $f_i$: file_8_view_0, activation_fee_cents for $i$
- $cat(i)$: file_7_view_0, category for $i$
- $loc(i)$: file_7_view_0, location_id for $i$
- $min_i$, $max_i$: file_7_view_0, minimum_lot, maximum_order for $i$
- $u_{i\ell}$: file_11_view_0, amount × 1000 if unit = GB, for matching item_ref and resource; 0 if missing
- $F_c$: file_3_view_0, activation_fee_cents for $c$
- $min_c$, $max_c$: file_3_view_0, minimum_quantity, maximum_quantity for $c$
- $cap_\ell$: sum of file_2_view_0, amount for resource = $\ell$
- $bonus_{ab}$: file_1_view_0, bonus_cents for $(a,b)$
- $P$: file_6_view_0, item_a, item_b
- $Q$: file_10_view_0, item_ref, prerequisite_ref

**Variable domains:**
- $x_i \in \{0\} \cup \{\text{integers}: min_i \leq x_i \leq max_i\}$
- $y_i, z_c, w_{ab} \in \{0,1\}$

**Maximize total net benefit in USD cents, subject to all above constraints.**