**Mathematical Model**

Let  

* \(P\) – set of product types (car models).  
* \(x_p\) – number of vehicles of type \(p\) ordered per day.  

Parameters  

* \(V_p\) – profit (value) obtained from selling one vehicle of type \(p\).  
* \(W_p\) – weight (or space) consumed by one vehicle of type \(p\).  
* \(C\) – total inventory capacity available per day.  

All variables are non‑negative integers.

\[
\begin{aligned}
\max\;& \sum_{p\in P} V_p\,x_p \\[4pt]
\text{s.t.}\;& \sum_{p\in P} W_p\,x_p \;\le\; C \\[4pt]
& x_p \in \mathbb{Z}_{\ge 0}\quad \forall p\in P
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source Table | Column(s) | Notes |
|--------|--------------|-----------|-------|
| \(P\) | `file_1_view_0` | `ProductName` | Each distinct `ProductName` defines an index element. |
| \(V_p\) | `file_1_view_0` | `Value` | Profit per unit of product \(p\). |
| \(W_p\) | `file_1_view_0` | `Weight` | Weight (capacity consumption) per unit of product \(p\). |
| \(C\) | `file_0_view_0` | `Capacity` | Overall daily inventory capacity. |

All mappings use the exact table identifiers and column names returned by **CSVQA**. No additional data or assumptions are introduced.