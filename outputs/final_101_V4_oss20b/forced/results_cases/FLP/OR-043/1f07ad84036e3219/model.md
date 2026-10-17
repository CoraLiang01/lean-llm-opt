**Mathematical Model**

Let  

* \(I=\{1,\dots ,n\}\) be the set of drug products ( \(n=20\) ).  
* \(v_i\) – benefit per unit of product \(i\).  
* \(w_i\) – weight (stock space) per unit of product \(i\).  
* \(C\) – total available stock capacity.  

Decision variable  

\[
x_i \ge 0 \qquad \forall i\in I
\]

Objective  

\[
\max \;\; \sum_{i\in I} v_i\,x_i
\]

Capacity constraint  

\[
\sum_{i\in I} w_i\,x_i \;\le\; C
\]

---

**Data Mapping**

| Table ID | Source Column | Parameter | Mapping Rule |
|----------|---------------|-----------|--------------|
| `file_0_view_0` | `Capacity` | \(C\) | `C = Capacity` (single value) |
| `file_1_view_0` | `ProductName` | index \(i\) | `i = source_row + 1` (1‑based) |
| `file_1_view_0` | `Value` | \(v_i\) | `v_i = Value` for each product row |
| `file_1_view_0` | `Weight` | \(w_i\) | `w_i = Weight` for each product row |

All parameters are taken directly from the CSV records; no additional numerical processing is performed. The model uses continuous, non‑negative decision variables \(x_i\).