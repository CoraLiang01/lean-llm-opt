## Symbolic Mathematical Model

### Sets
- $I$: Set of item options (indexed by $i$), each identified by $(\text{item\_ref}_i, \text{location}_i)$ from item options tables (file_6_view_0 and file_7_view_0).
- $S$: Set of display sections (resources) (from resource capacity ledger, e.g., SECTION_A, SECTION_B, SECTION_C).
- $C$: Set of categories (from category limits and fees, file_3_view_0).
- $B$: Set of bundle pairs $(i,j)$ with bonus (from bundle bonuses, file_1_view_0).
- $F$: Set of incompatible pairs $(i,j)$ (from incompatible pairs, file_5_view_0).
- $R$: Set of requires pairs $(i,j)$ (from requires pairs, file_10_view_0).

### Parameters
- $b_i$: Per-unit net benefit (sum of amount_cents for all components of $i$; file_0_view_0).
- $f_i$: Fixed item fee (activation_fee_cents for $i$; file_8_view_0).
- $a_i$: Authorization indicator (1 if authorized, 0 if not; from item options).
- $l_i$: Minimum lot size for $i$ (minimum_lot; from item options).
- $u_i$: Maximum order for $i$ (maximum_order; from item options).
- $cat_i$: Category of $i$ (from item options).
- $sec_i$: Section/resource of $i$ (location_id; from item options).
- $q_{i,s}$: Per-pack usage of resource $s$ by $i$ (in ml; sum all usage rows for $i$ and $s$ from file_11_view_0 and file_12_view_0, converting liters to ml).
- $cap_s$: Total available capacity for section/resource $s$ (sum of amount for $s$ in resource capacity ledger, file_2_view_0).
- $catmin_c$, $catmax_c$: Category $c$ lower/upper quantity limits (from file_3_view_0).
- $catfee_c$: Category activation fee (from file_3_view_0).
- $bonus_{ij}$: Bundle bonus for pair $(i,j)$ (from file_1_view_0).
- $I_c$: Set of items in category $c$.
- $I_s$: Set of items in section/resource $s$.

### Decision Variables
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of packs of item $i$ to display.
- $y_i \in \{0,1\}$: 1 if $x_i > 0$, 0 otherwise (item activation).
- $z_c \in \{0,1\}$: 1 if any $x_i > 0$ for $i \in I_c$ (category activation).
- $w_{ij} \in \{0,1\}$: 1 if both $x_i > 0$ and $x_j > 0$ (bundle activation).

### Objective
Maximize net merchandising benefit in USD cents:
\[
\max \Bigg[
\sum_{i \in I} b_i x_i
- \sum_{i \in I} f_i y_i
- \sum_{c \in C} catfee_c \cdot z_c
+ \sum_{(i,j) \in B} bonus_{ij} w_{ij}
\Bigg]
\]

### Constraints

#### 1. Authorization and Order Bounds
\[
x_i = 0 \quad \forall i \text{ with } a_i = 0
\]
\[
x_i \in \{0\} \cup [l_i, u_i] \cap \mathbb{Z} \quad \forall i \text{ with } a_i = 1
\]

#### 2. Section/Resource Capacity
\[
\sum_{i \in I_s} q_{i,s} x_i \leq cap_s \quad \forall s \in S
\]

#### 3. Category Quantity Limits
\[
catmin_c \leq \sum_{i \in I_c} x_i \leq catmax_c \quad \forall c \in C
\]

#### 4. Item Activation
\[
y_i = 
\begin{cases}
1 & \text{if } x_i \geq l_i \\
0 & \text{if } x_i = 0
\end{cases}
\quad \forall i \in I
\]
(Enforced via: $x_i \geq l_i y_i$, $x_i \leq u_i y_i$)

#### 5. Category Activation
\[
z_c \geq y_i \quad \forall i \in I_c, \forall c \in C
\]
\[
z_c \leq \sum_{i \in I_c} y_i \leq |I_c| z_c \quad \forall c \in C
\]
(Or simply $z_c = \min(1, \sum_{i \in I_c} y_i)$)

#### 6. Incompatible Pairs
\[
y_i + y_j \leq 1 \quad \forall (i,j) \in F
\]

#### 7. Requires Pairs
\[
y_i \leq y_j \quad \forall (i,j) \in R
\]

#### 8. Bundle Bonuses
\[
w_{ij} \leq y_i, \quad w_{ij} \leq y_j, \quad w_{ij} \geq y_i + y_j - 1 \quad \forall (i,j) \in B
\]

#### 9. Variable Domains
\[
x_i \in \{0\} \cup [l_i, u_i] \cap \mathbb{Z} \quad \forall i \text{ with } a_i = 1
\]
\[
y_i \in \{0,1\} \quad \forall i \in I
\]
\[
z_c \in \{0,1\} \quad \forall c \in C
\]
\[
w_{ij} \in \{0,1\} \quad \forall (i,j) \in B
\]

---

### Data Mapping

- **file_0_view_0**: Per-unit benefit $b_i$ for each $i$ (sum all amount_cents for each item_ref).
- **file_1_view_0**: Bundle pairs $B$ and bonus values $bonus_{ij}$.
- **file_2_view_0**: Section/resource capacities $cap_s$ (sum of amount for each resource).
- **file_3_view_0**: Category limits $catmin_c$, $catmax_c$, and activation fees $catfee_c$.
- **file_4_view_0**: Item identity (item_ref, entity_id, display_name).
- **file_5_view_0**: Incompatible pairs $F$.
- **file_6_view_0** and **file_7_view_0**: Item options $I$ with fields: item_ref, location_id (section), category, authorized, minimum_lot $l_i$, maximum_order $u_i$.
- **file_8_view_0**: Item fixed fees $f_i$.
- **file_10_view_0**: Requires pairs $R$.
- **file_11_view_0** and **file_12_view_0**: Item usage $q_{i,s}$ (convert all 'liter' to ml by multiplying by 1000).
- All indices and sets are defined by the union of relevant rows in the above tables, filtered to the three display sections of MARKET_SQUARE.

---

**All constraints, sets, and parameters are mapped directly from the supplied tables as described above.**