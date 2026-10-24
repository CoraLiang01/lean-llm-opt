**Mathematical Model**

Let  

* \(I\) – set of all `item_ref` values in **file_7_view_0**  
* \(C\) – set of all `category` values in **file_3_view_0**  
* \(R\) – set of all `resource` values in **file_2_view_0**  
* \(B\) – set of all bundle rows in **file_1_view_0**  

Parameters  

| Symbol | Definition | Source table & column |
|--------|------------|-----------------------|
| \(p_i\) | Benefit per unit (USD cents) for item \(i\) | \(\displaystyle p_i=\sum_{b\in\text{benefit},\,b.item_ref=i}\frac{b.amount\; \text{fx\_num}_{b.currency}}{\text{fx\_den}_{b.currency}}\) – **file_0_view_0** & **file_4_view_0** |
| \(\text{item\_fee}_i\) | One‑time fee for item \(i\) | **file_8_view_0** (`activation_fee_cents`) |
| \(\text{cap}_r\) | Capacity of resource \(r\) (MB) | **file_2_view_0** (`amount`) |
| \(u_{i,r}\) | Usage of resource \(r\) per unit of item \(i\) (MB) | **file_11_view_0** (`amount`) |
| \(\text{min\_lot}_i,\;\text{max\_order}_i\) | Minimum lot and maximum order for item \(i\) | **file_7_view_0** (`minimum_lot`,`maximum_order`) |
| \(\text{authorized}_i\) | 1 if item \(i\) is authorized, 0 otherwise | **file_7_view_0** (`authorized`) |
| \(\text{cat}_i\) | Category of item \(i\) | **file_7_view_0** (`category`) |
| \(\text{min\_q}_c,\;\text{max\_q}_c\) | Minimum and maximum quantity for category \(c\) | **file_3_view_0** (`minimum_quantity`,`maximum_quantity`) |
| \(\text{activation\_fee}_c\) | Activation fee for category \(c\) | **file_3_view_0** (`activation_fee_cents`) |
| \(\text{bonus}_b\) | Bonus (USD cents) for bundle \(b\) | **file_1_view_0** (`bonus_cents`) |
| \((i,j)\in\text{Incomp}\) | Incompatible pair of items | **file_6_view_0** (`item_a`,`item_b`) |
| \((i,j)\in\text{Req}\) | Requires pair: item \(i\) requires item \(j\) | **file_10_view_0** (`item_ref`,`prerequisite_ref`) |

Decision variables  

* \(x_i \in \mathbb{Z}_+\) – quantity of item \(i\) (units)  
* \(y_i \in \{0,1\}\) – 1 if item \(i\) is selected (quantity > 0)  
* \(z_c \in \{0,1\}\) – 1 if category \(c\) is used  
* \(w_b \in \{0,1\}\) – 1 if both items in bundle \(b\) are selected  

**Objective**  

\[
\max \;\sum_{i\in I} p_i\,x_i
      \;-\;\sum_{i\in I} \text{item\_fee}_i\,y_i
      \;-\;\sum_{c\in C} \text{activation\_fee}_c\,z_c
      \;+\;\sum_{b\in B} \text{bonus}_b\,w_b
\]

**Constraints**

1. **Quantity bounds**  
   \[
   \text{min\_lot}_i\,y_i \;\le\; x_i \;\le\; \text{max\_order}_i\,y_i
   \quad \forall i\in I
   \]
2. **Authorization**  
   \[
   y_i \;\le\; \text{authorized}_i
   \quad \forall i\in I
   \]
3. **Capacity**  
   \[
   \sum_{i\in I} u_{i,r}\,x_i \;\le\; \text{cap}_r
   \quad \forall r\in R
   \]
4. **Category quantity limits**  
   \[
   \text{min\_q}_c \;\le\; \sum_{i:\text{cat}_i=c} x_i
   \;\le\; \text{max\_q}_c
   \quad \forall c\in C
   \]
5. **Category activation**  
   \[
   z_c \;\ge\; y_i \quad \forall i:\text{cat}_i=c,\;\forall c\in C
   \]
6. **Incompatibility**  
   \[
   y_i + y_j \;\le\; 1 \quad \forall (i,j)\in\text{Incomp}
   \]
7. **Requires**  
   \[
   y_i \;\le\; y_j \quad \forall (i,j)\in\text{Req}
   \]
8. **Bundle activation**  
   \[
   w_b \;\le\; y_{i(b)} ,\;
   w_b \;\le\; y_{j(b)} ,\;
   w_b \;\ge\; y_{i(b)} + y_{j(b)} - 1
   \quad \forall b\in B
   \]
9. **Variable domains**  
   \[
   x_i \in \mathbb{Z}_+,\; y_i,z_c,w_b \in \{0,1\}
   \quad \forall i\in I,\;c\in C,\;b\in B
   \]

---

**Data Mapping Rules**

| Index / Parameter | Source table | Source columns |
|-------------------|--------------|----------------|
| \(I\) | **file_7_view_0** | `item_ref` |
| \(C\) | **file_3_view_0** | `category` |
| \(R\) | **file_2_view_0** | `resource` |
| \(B\) | **file_1_view_0** | `item_a`, `item_b` |
| \(p_i\) | **file_0_view_0** (`amount`), **file_4_view_0** (`usd_cents_numerator`,`denominator`) | join on `currency` |
| \(\text{item\_fee}_i\) | **file_8_view_0** | `activation_fee_cents` |
| \(\text{cap}_r\) | **file_2_view_0** | `amount` |
| \(u_{i,r}\) | **file_11_view_0** | `amount` |
| \(\text{min\_lot}_i,\;\text{max\_order}_i\) | **file_7_view_0** | `minimum_lot`,`maximum_order` |
| \(\text{authorized}_i\) | **file_7_view_0** | `authorized` |
| \(\text{cat}_i\) | **file_7_view_0** | `category` |
| \(\text{min\_q}_c,\;\text{max\_q}_c\) | **file_3_view_0** | `minimum_quantity`,`maximum_quantity` |
| \(\text{activation\_fee}_c\) | **file_3_view_0** | `activation_fee_cents` |
| \(\text{bonus}_b\) | **file_1_view_0** | `bonus_cents` |
| \((i,j)\in\text{Incomp}\) | **file_6_view_0** | `item_a`,`item_b` |
| \((i,j)\in\text{Req}\) | **file_10_view_0** | `item_ref`,`prerequisite_ref` |

All numeric values are taken directly from the returned rows; no additional aggregation or conversion is performed beyond the explicit unit equivalence (1000 MB = 1 GB) already applied in the unit‑prepared view. The model uses only the data present in the current observation.