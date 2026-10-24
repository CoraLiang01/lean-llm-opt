**Mathematical Model**

Let  

- \(I\) be the set of drug products.  
- For each \(i\in I\):  
  - \(c_i\) – benefit (value) per unit of product \(i\).  
  - \(w_i\) – weight (capacity consumption) per unit of product \(i\).  
- \(C\) – total available stock capacity.

Decision variables  

\[
x_i \;\ge\; 0 , \qquad \forall i\in I
\]

Objective  

\[
\max \;\sum_{i\in I} c_i\,x_i
\]

Capacity constraint  

\[
\sum_{i\in I} w_i\,x_i \;\le\; C
\]

---

**Data Mapping Rules**

| Parameter | Source | Table ID | Column(s) | Notes |
|-----------|--------|----------|-----------|-------|
| \(C\) | Overall capacity | `file_0_view_0` | `Capacity` | Single value |
| \(c_i\) | Benefit per unit | `file_1_view_0` | `Value` | Indexed by `ProductName` |
| \(w_i\) | Weight per unit | `file_1_view_0` | `Weight` | Indexed by `ProductName` |
| \(I\) | Product set | `file_1_view_0` | `ProductName` | Each distinct product name forms an index element |

All indices, parameters, and variables are defined exactly as above; no additional data or constraints are introduced.