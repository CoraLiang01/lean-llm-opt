## Symbolic Mathematical Model

**Sets:**
- $I$: set of item options (item_ref from file_7_view_0)
- $L$: set of platforms (location_id from file_7_view_0; e.g., PC, CONSOLE, MOBILE)
- $C$: set of categories (category from file_3_view_0)
- $R$: set of resources (resource from file_2_view_0; e.g., PC, CONSOLE, MOBILE)
- $B$: set of bundle pairs (rows in file_1_view_0)
- $Q$: set of incompatible pairs (rows in file_6_view_0)
- $P$: set of requires pairs (rows in file_10_view_0)

**Parameters:**
- $a_i$: authorized flag for item $i$ (from file_7_view_0, column authorized)
- $l_i$: platform of item $i$ (from file_7_view_0, column location_id)
- $c_i$: category of item $i$ (from file_7_view_0, column category)
- $q^{\min}_i$, $q^{\max}_i$: minimum_lot and maximum_order for item $i$ (from file_7_view_0)
- $b_{i}$: per-unit benefit in USD cents for item $i$ (sum over all components in file_0_view_0 for $i$, each converted to USD cents using file_4_view_0)
- $f^{\text{item}}_i$: item_fee for item $i$ (from file_8_view_0, activation_fee_cents)
- $u_{i,r}$: resource usage per unit of item $i$ for resource $r$ (from file_11_view_0, amount × 1000 if unit is GB, else as is; 0 if missing)
- $K_r$: total available resource $r$ (sum of amount in file_2_view_0 for resource $r$)
- $C_c^{\min}$, $C_c^{\max}$: minimum and maximum quantity for category $c$ (from file_3_view_0)
- $f^{\text{cat}}_c$: activation_fee_cents for category $c$ (from file_3_view_0)
- $S_i$: set of items in category $c_i$
- $b_{ij}$: bundle bonus_cents for bundle $(i,j)$ (from file_1_view_0)
- $Q$: set of incompatible pairs $(i,j)$ (from file_6_view_0)
- $P$: set of requires pairs $(i,k)$ (from file_10_view_0; $i$ requires $k$)

**Decision Variables:**
- $x_i \in \mathbb{Z}_+$: quantity of item $i$ selected ($x_i = 0$ if not selected)
- $y_i \in \{0,1\}$: 1 if $x_i > 0$, 0 otherwise (item activation indicator)
- $z_c \in \{0,1\}$: 1 if any $x_i > 0$ for $i$ in category $c$ (category activation indicator)
- $w_{ij} \in \{0,1\}$: 1 if both $x_i > 0$ and $x_j > 0$ (bundle activation indicator)

**Objective:**
Maximize net benefit in USD cents:
\[
\max \Bigg\{ 
\sum_{i \in I} \left[ b_i x_i - f^{\text{item}}_i y_i \right]
+ \sum_{(i,j) \in B} b_{ij} w_{ij}
- \sum_{c \in C} f^{\text{cat}}_c z_c
\Bigg\}
\]

**Constraints:**

1. **Authorization and Order Bounds:**
   \[
   x_i = 0 \quad \forall i \in I \text{ with } a_i = 0
   \]
   \[
   x_i = 0 \text{ or } q^{\min}_i \leq x_i \leq q^{\max}_i \quad \forall i \in I \text{ with } a_i = 1
   \]
   \[
   x_i \in \mathbb{Z}_+ \quad \forall i \in I
   \]

2. **Item Activation:**
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]
   \[
   x_i \leq q^{\max}_i y_i \quad \forall i \in I
   \]
   \[
   x_i \geq q^{\min}_i y_i \quad \forall i \in I \text{ with } a_i = 1
   \]
   \[
   x_i = 0 \implies y_i = 0
   \]

3. **Category Quantity and Activation:**
   \[
   C_c^{\min} \leq \sum_{i \in I: c_i = c} x_i \leq C_c^{\max} \quad \forall c \in C
   \]
   \[
   z_c \in \{0,1\} \quad \forall c \in C
   \]
   \[
   x_i \leq q^{\max}_i z_{c_i} \quad \forall i \in I
   \]

4. **Resource Capacity (per platform):**
   \[
   \sum_{i \in I} u_{i,r} x_i \leq K_r \quad \forall r \in R
   \]

5. **Incompatibility:**
   \[
   y_i + y_j \leq 1 \quad \forall (i,j) \in Q
   \]

6. **Requires:**
   \[
   y_i \leq y_k \quad \forall (i,k) \in P
   \]

7. **Bundle Bonuses:**
   \[
   w_{ij} \leq y_i, \quad w_{ij} \leq y_j, \quad w_{ij} \geq y_i + y_j - 1 \quad \forall (i,j) \in B
   \]
   \[
   w_{ij} \in \{0,1\} \quad \forall (i,j) \in B
   \]

**Data Mapping:**
- $I$: All item_ref in file_7_view_0
- $L$: All location_id in file_7_view_0
- $C$: All category in file_3_view_0
- $R$: All resource in file_2_view_0
- $B$: All (item_a, item_b) in file_1_view_0
- $Q$: All (item_a, item_b) in file_6_view_0
- $P$: All (item_ref, prerequisite_ref) in file_10_view_0
- $a_i$: file_7_view_0.authorized
- $l_i$: file_7_view_0.location_id
- $c_i$: file_7_view_0.category
- $q^{\min}_i$: file_7_view_0.minimum_lot
- $q^{\max}_i$: file_7_view_0.maximum_order
- $b_i$: sum over file_0_view_0.amount for item_ref $i$, each converted to USD cents using file_4_view_0 (currency, usd_cents_numerator, denominator)
- $f^{\text{item}}_i$: file_8_view_0.activation_fee_cents
- $u_{i,r}$: file_11_view_0.amount × 1000 if unit is GB, else as is, for item_ref $i$ and resource $r$; 0 if missing
- $K_r$: sum of file_2_view_0.amount for resource $r$
- $C_c^{\min}$, $C_c^{\max}$: file_3_view_0.minimum_quantity, maximum_quantity for category $c$
- $f^{\text{cat}}_c$: file_3_view_0.activation_fee_cents
- $b_{ij}$: file_1_view_0.bonus_cents for bundle $(i,j)$

**Variable Domains:**
- $x_i \in \{0\} \cup [q^{\min}_i, q^{\max}_i] \cap \mathbb{Z}$ for $a_i = 1$; $x_i = 0$ for $a_i = 0$
- $y_i, z_c, w_{ij} \in \{0,1\}$

**Notes:**
- All sums, indices, and mappings are over the current rows of the respective tables.
- All units are in USD cents, MB, and integer package counts as per the data.
- All constraints and parameters are mapped directly from the supplied tables as described above.