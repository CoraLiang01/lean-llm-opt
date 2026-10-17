**Mathematical Model**

Let  

* \(P\) – set of all products (identified by *Product Name*).  
* For each \(p\in P\)  
  * \(r_p\) – revenue per unit (from column **Revenue**).  
  * \(d_p\) – known demand (from column **Demand**).  
  * \(s_p\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

\[
x_p \;\ge\; 0 \qquad \forall p\in P
\]

(continuous quantity of product \(p\) to sell).

---

**Objective**

\[
\max \;\sum_{p\in P} r_p\,x_p
\]

---

**Constraints**

\[
\begin{aligned}
x_p &\;\le\; d_p &&\forall p\in P &&\text{(cannot sell more than demand)}\\[4pt]
x_p &\;\le\; s_p &&\forall p\in P &&\text{(cannot sell more than inventory)}\\[4pt]
x_p &\;\ge\; 0   &&\forall p\in P
\end{aligned}
\]

---

### Data Mapping Rules

| Parameter | Source Table | Source Column | Mapping Rule |
|-----------|--------------|---------------|--------------|
| \(P\) | `file_0_view_0` | `Product Name` | Each distinct value forms an index in \(P\). |
| \(r_p\) | `file_0_view_0` | `Revenue` | \(r_p =\) value of **Revenue** for product \(p\). |
| \(d_p\) | `file_0_view_0` | `Demand` | \(d_p =\) value of **Demand** for product \(p\). |
| \(s_p\) | `file_0_view_0` | `Initial Inventory` | \(s_p =\) value of **Initial Inventory** for product \(p\). |
| \(x_p\) | – | – | Decision variable defined for each \(p\in P\). |

All numeric values are parsed as floating‑point numbers; identifiers are treated as categorical strings. No additional constraints or parameters are introduced beyond those specified.