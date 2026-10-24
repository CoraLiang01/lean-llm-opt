**Mathematical Model**

Let  

* \(P=\{1,\dots ,80\}\) – set of products  
* \(D=\{1,\dots ,22\}\) – set of days in the month  

**Parameters**  

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(d_p\) | Maximum demand of product \(p\) (units of 100 kg) | `file_0_view_0`, row 0, column `A{p}` |
| \(p_p\) | Selling price of product \(p\) ($/100 kg) | `file_0_view_0`, row 1, column `A{p}` |
| \(c_p\) | Production cost of product \(p\) ($/100 kg) | `file_0_view_0`, row 2, column `A{p}` |
| \(q_p\) | Daily production quota of product \(p\) (max units per day) | `file_0_view_0`, row 3, column `A{p}` |
| \(m_p\) | Minimum batch size of product \(p\) (units of 100 kg) | `file_2_view_0`, row 0, column `A{p}` |
| \(a_p\) | Activation cost of product \(p\) ($) | `file_1_view_0`, row 0, column `A{p}` |

**Decision Variables**

* \(x_{p,d}\in \mathbb{Z}_{\ge 0}\) – quantity of product \(p\) produced on day \(d\) (units of 100 kg)  
* \(z_p\in\{0,1\}\) – 1 if product \(p\) is produced at all, 0 otherwise  

**Objective**

\[
\max \;\sum_{p\in P}\sum_{d\in D}\bigl(p_p-c_p\bigr)\,x_{p,d}\;-\;\sum_{p\in P} a_p\,z_p
\]

**Constraints**

1. **Daily quota**  
   \[
   x_{p,d}\;\le\; q_p \quad \forall p\in P,\; d\in D
   \]

2. **Demand limit**  
   \[
   \sum_{d\in D} x_{p,d}\;\le\; d_p \quad \forall p\in P
   \]

3. **Batch‑size requirement**  
   \[
   \sum_{d\in D} x_{p,d}\;\ge\; m_p\,z_p \quad \forall p\in P
   \]

4. **Activation linkage (big‑M)**  
   \[
   \sum_{d\in D} x_{p,d}\;\le\; d_p\,z_p \quad \forall p\in P
   \]

5. **Variable domains**  
   \[
   x_{p,d}\in \mathbb{Z}_{\ge 0}\quad \forall p,d;\qquad
   z_p\in\{0,1\}\quad \forall p
   \]

---

**Data Mapping**

```json
{
  "parameters": {
    "demand": {"table_id":"file_0_view_0","row":0,"column_prefix":"A"},
    "price":  {"table_id":"file_0_view_0","row":1,"column_prefix":"A"},
    "cost":   {"table_id":"file_0_view_0","row":2,"column_prefix":"A"},
    "quota":  {"table_id":"file_0_view_0","row":3,"column_prefix":"A"},
    "min_batch": {"table_id":"file_2_view_0","row":0,"column_prefix":"A"},
    "activation_cost": {"table_id":"file_1_view_0","row":0,"column_prefix":"A"}
  },
  "index_sets": {
    "P": {"size":80,"ids":"1..80"},
    "D": {"size":22,"ids":"1..22"}
  }
}
```

This symbolic formulation, together with the explicit data mapping, fully specifies the integer programming model for maximizing profit under the given production, demand, quota, batch‑size, and activation‑cost constraints.