**Mathematical Model**

Let  

- \(I=\{i\}\) be the set of real‑estate areas (ProductName).  
- \(v_i\) be the benefit per unit of development in area \(i\).  
- \(w_i\) be the weight (resource consumption) per unit of development in area \(i\).  
- \(C\) be the overall development capacity.  

Decision variable  

\[
x_i \;\ge\; 0 \qquad \forall i\in I
\]

Objective  

\[
\max \;\sum_{i\in I} v_i\,x_i
\]

Capacity constraint  

\[
\sum_{i\in I} w_i\,x_i \;\le\; C
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Source Column | Mapping Rule |
|-----------|--------------|---------------|--------------|
| \(C\) | `file_0_view_0` | `Capacity` | Single value → parameter \(C\) |
| \(I\) | `file_1_view_0` | `ProductName` | Each distinct `ProductName` → index \(i\) |
| \(v_i\) | `file_1_view_0` | `Value` | For each record, map `Value` to \(v_i\) |
| \(w_i\) | `file_1_view_0` | `Weight` | For each record, map `Weight` to \(w_i\) |

These mappings bind every parameter and index to the corresponding source data without any aggregation or omission.