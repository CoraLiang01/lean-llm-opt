**Symbolic Model**

Let  

* \(P=\{\,\text{Widget}1,\dots ,\text{Widget}141\,\}\) – set of products.  
* \(x_p\ge 0\) – units of product \(p\in P\) produced.  
* \(s\ge 0\) – kilograms of CatalystX sold.  
* \(d\ge 0\) – kilograms of CatalystX disposed.

Parameters (all obtained from the CSV files)

| Symbol | Source | Mapping |
|--------|--------|---------|
| \(L^{\text{LH}}\) | `resource_limits.csv` | value of **MonthlyLimit** where **Resource** = *LaborHours* |
| \(L^{\text{A}}\) | `resource_limits.csv` | value of **MonthlyLimit** where **Resource** = *MaterialA* |
| \(L^{\text{B}}\) | `resource_limits.csv` | value of **MonthlyLimit** where **Resource** = *MaterialB* |
| \(c^{\text{LH}}_p\) | `product_resources.csv` | **LaborHours** for product \(p\) |
| \(c^{\text{A}}_p\) | `product_resources.csv` | **MaterialA** for product \(p\) |
| \(c^{\text{B}}_p\) | `product_resources.csv` | **MaterialB** for product \(p\) |
| \(\pi_p\) | `product_resources.csv` | **Profit** for product \(p\) |
| \(g\) | fixed | 5 (kg CatalystX per unit of *Widget3*) |
| \(S_{\max}\) | fixed | 1500 (kg) |
| \(p_s\) | fixed | 300 (USD/kg) |
| \(p_d\) | fixed | 200 (USD/kg) |

---

### Objective

\[
\max \;\; \sum_{p\in P}\pi_p\,x_p \;+\; p_s\,s \;-\; p_d\,d
\]

### Constraints

1. **Labor hours**  
   \[
   \sum_{p\in P} c^{\text{LH}}_p\,x_p \;\le\; L^{\text{LH}}
   \]

2. **Material A**  
   \[
   \sum_{p\in P} c^{\text{A}}_p\,x_p \;\le\; L^{\text{A}}
   \]

3. **Material B**  
   \[
   \sum_{p\in P} c^{\text{B}}_p\,x_p \;\le\; L^{\text{B}}
   \]

4. **CatalystX generation**  
   \[
   d \;=\; g\,x_{\text{Widget3}} \;-\; s
   \]

5. **CatalystX sale cap**  
   \[
   s \;\le\; S_{\max}
   \]

6. **Non‑negativity**  
   \[
   x_p \ge 0 \;\;\forall p\in P,\qquad s\ge 0,\qquad d\ge 0
   \]

---

**Data Mapping Rules**

* `file_0_view_0` (resource_limits.csv)  
  * `Resource` column → index for \(L^{\text{LH}}, L^{\text{A}}, L^{\text{B}}\)  
  * `MonthlyLimit` column → corresponding limit value.

* `file_1_view_0` (product_resources.csv)  
  * `Product` column → index \(p\in P\).  
  * `LaborHours` → \(c^{\text{LH}}_p\).  
  * `MaterialA` → \(c^{\text{A}}_p\).  
  * `MaterialB` → \(c^{\text{B}}_p\).  
  * `Profit` → \(\pi_p\).

* Fixed constants \(g=5\), \(S_{\max}=1500\), \(p_s=300\), \(p_d=200\) are hard‑coded.  

This formulation captures the production‑planning problem with CatalystX by‑product, respecting all resource limits, sale cap, and disposal cost.