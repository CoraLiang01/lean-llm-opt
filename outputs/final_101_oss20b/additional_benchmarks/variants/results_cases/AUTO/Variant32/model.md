**Mathematical Model**

Let  

* \(I\) – set of items (item\_ref) that survive the selection rule.  
* \(C\) – set of categories that survive the selection rule.  
* \(B\) – set of bundles that survive the selection rule.  
* \(R\) – set of resources that appear in the usage table after filtering.  

For each item \(i\in I\) let  

* \(x_i\in\mathbb Z_{\ge0}\) – quantity ordered (units of 100 kg).  
* \(z_i\in\{0,1\}\) – indicator that item \(i\) is ordered (positive quantity).  

For each category \(c\in C\) let  

* \(y_c\in\{0,1\}\) – indicator that the category is used.  

For each bundle \(b\in B\) let  

* \(w_b\in\{0,1\}\) – indicator that the bundle bonus is earned.  

Parameters (all integers, USD cents unless otherwise noted)

| Symbol | Meaning | Source table & columns |
|--------|---------|------------------------|
| \(\text{benefit}_i\) | Net benefit per unit of item \(i\) (sum of all benefit records, converted to USD cents) | `benefit` (item_ref, amount, currency) + `fx` (currency → usd_cents_numerator/denominator) |
| \(\text{item\_fee}_i\) | Item‑fee for item \(i\) (USD cents) | `item_fee` (item_ref, activation_fee_cents) |
| \(\text{cat\_fee}_c\) | Category activation fee for category \(c\) (USD cents) | `category` (category, activation_fee_cents) |
| \(\text{bonus}_b\) | Bundle bonus for bundle \(b\) (USD cents) | `bundle` (bonus_cents) |
| \(\text{usage}_{i,r}\) | Resource consumption per unit of item \(i\) for resource \(r\) (converted units) | `usage` (item_ref, resource, amount) |
| \(\text{cap}_r\) | Signed capacity ledger total for resource \(r\) (converted units) | `capacity_ledger` (resource, amount) |
| \(\text{min\_lot}_i,\text{max\_order}_i\) | Minimum and maximum order limits for item \(i\) | `item` (minimum_lot, maximum_order) |
| \(\text{auth}_i\in\{0,1\}\) | Authorization flag for item \(i\) (1 if authorized, 0 otherwise) | `item` (authorized) |
| \(\text{cat}_i\) | Category of item \(i\) | `item` (category) |
| \(\text{a}_b,\text{b}_b\) | Items in bundle \(b\) | `bundle` (item_a, item_b) |
| \((i,j)\in P\) | Incompatible pair of items | `incompatible` (item_a, item_b) |
| \((i,j)\in Q\) | Requires relation: item \(i\) requires item \(j\) | `requires` (item_ref, prerequisite_ref) |

All tables are first filtered by  

* `dealership_id = 'OSLO_NEW_CARS'`  
* `effective_date ≤ 2026‑05‑07`  

and then the **selection rule** is applied: for each key \((\text{dealership_id},\text{table},\text{record_id})\) keep the record with the highest integer revision; discard the record if that revision has `action = 'DELETE'`. Identical retransmissions are collapsed to one record.

---

### Decision Variables

\[
\begin{aligned}
x_i &\in \mathbb Z_{\ge 0} &&\forall i\in I\\
z_i &\in \{0,1\} &&\forall i\in I\\
y_c &\in \{0,1\} &&\forall c\in C\\
w_b &\in \{0,1\} &&\forall b\in B
\end{aligned}
\]

---

### Constraints

1. **Order limits (authorized items only)**  
   \[
   \begin{aligned}
   x_i &\ge \text{min\_lot}_i \; z_i &&\forall i\in I\\
   x_i &\le \text{max\_order}_i \; z_i &&\forall i\in I\\
   z_i &\le \text{auth}_i &&\forall i\in I
   \end{aligned}
   \]

2. **Category activation**  
   \[
   \begin{aligned}
   y_c &\ge z_i &&\forall c\in C,\; i\in I:\text{cat}_i=c\\
   y_c &\le \sum_{i:\text{cat}_i=c} z_i &&\forall c\in C
   \end{aligned}
   \]

3. **Incompatibilities**  
   \[
   z_i + z_j \le 1 &&\forall (i,j)\in P
   \]

4. **Requires dependencies**  
   \[
   z_i \le z_j &&\forall (i,j)\in Q
   \]

5. **Bundle bonuses**  
   \[
   \begin{aligned}
   w_b &\le z_{\text{a}_b} &&\forall b\in B\\
   w_b &\le z_{\text{b}_b} &&\forall b\in B\\
   w_b &\ge z_{\text{a}_b} + z_{\text{b}_b} - 1 &&\forall b\in B
   \end{aligned}
   \]

6. **Resource capacities**  
   \[
   \sum_{i\in I} x_i \, \text{usage}_{i,r} \;\le\; \text{cap}_r
   &&\forall r\in R
   \]

All variables are integer (binary for \(z_i, y_c, w_b\)).

---

### Objective

\[
\max \;\;
\sum_{i\in I} x_i \, \text{benefit}_i
\;-\;
\sum_{i\in I} z_i \, \text{item\_fee}_i
\;-\;
\sum_{c\in C} y_c \, \text{cat\_fee}_c
\;+\;
\sum_{b\in B} w_b \, \text{bonus}_b
\]

The objective value is the maximum net benefit in USD cents.

---

**Data‑Mapping Summary**

| Set / Parameter | Source table | Key columns | Notes |
|-----------------|--------------|-------------|-------|
| \(I\) | `item` | `item_ref` | After filtering |
| \(C\) | `category` | `category` | After filtering |
| \(B\) | `bundle` | `record_id` | After filtering |
| \(R\) | `usage` | `resource` | After filtering |
| \(\text{benefit}_i\) | `benefit` + `fx` | `item_ref`, `amount`, `currency` | Sum over selected records, convert via `fx` |
| \(\text{item\_fee}_i\) | `item_fee` | `item_ref`, `activation_fee_cents` | After filtering |
| \(\text{cat\_fee}_c\) | `category` | `category`, `activation_fee_cents` | After filtering |
| \(\text{bonus}_b\) | `bundle` | `bonus_cents` | After filtering |
| \(\text{usage}_{i,r}\) | `usage` | `item_ref`, `resource`, `amount` | Units match the resource |
| \(\text{cap}_r\) | `capacity_ledger` | `resource`, `amount` | Sum over selected records |
| \(\text{min\_lot}_i,\text{max\_order}_i\) | `item` | `minimum_lot`, `maximum_order` | After filtering |
| \(\text{auth}_i\) | `item` | `authorized` | 1 if non‑zero, else 0 |
| \(\text{cat}_i\) | `item` | `category` | After filtering |
| \(\text{a}_b,\text{b}_b\) | `bundle` | `item_a`, `item_b` | After filtering |
| \(P\) | `incompatible` | `item_a`, `item_b` | After filtering |
| \(Q\) | `requires` | `item_ref`, `prerequisite_ref` | After filtering |

All numeric values are integers; resource units are ml, minute, wh. The model respects the exact selection rule, unit conversions, and all stated constraints.