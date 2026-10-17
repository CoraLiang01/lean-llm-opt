**Mathematical Model**

Let  

- \(I\) be the set of drug types (indexed by the product names).  
- For each \(i\in I\)  
  - \(v_i\) = benefit coefficient (value) of drug \(i\).  
  - \(w_i\) = weight of one unit of drug \(i\).  
- \(C\) = overall inventory capacity.  

Decision variables  

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

Objective  

\[
\max \;\; \sum_{i\in I} v_i\,x_i
\]

Capacity constraint  

\[
\sum_{i\in I} w_i\,x_i \;\le\; C
\]

No other constraints are required.

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(C\) | `file_0_view_0` | `Capacity` |
| \(v_i\) | `file_1_view_0` | `Value` |
| \(w_i\) | `file_1_view_0` | `Weight` |
| Index set \(I\) | `file_1_view_0` | `ProductName` |

All parameter values are taken directly from the corresponding columns of the specified tables; no additional data or transformations are introduced.