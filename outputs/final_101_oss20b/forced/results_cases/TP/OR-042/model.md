**Mathematical Model**

Let  

- \(I\) be the set of product names extracted from **file_1_view_0**.  
- For each \(i\in I\)  
  - \(v_i\) = benefit coefficient (column **Value**)  
  - \(w_i\) = weight per unit (column **Weight**)  
- \(C\) = overall inventory capacity (column **Capacity** from **file_0_view_0**).  

Decision variables  

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

Objective  

\[
\max \;\sum_{i\in I} v_i\,x_i
\]

Capacity constraint  

\[
\sum_{i\in I} w_i\,x_i \;\le\; C
\]

All other constraints are the non‑negativity and integrality of \(x_i\).

---

**Data Mapping Rules**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(v_i\)   | file_1_view_0 | Value |
| \(w_i\)   | file_1_view_0 | Weight |
| \(C\)     | file_0_view_0 | Capacity |

The index set \(I\) is the set of distinct **ProductName** values in **file_1_view_0**. No other data are required.