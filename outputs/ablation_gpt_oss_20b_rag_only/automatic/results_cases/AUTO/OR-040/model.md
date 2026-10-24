**Mathematical Model**

Let  

- \(I\) be the set of area names (ProductName) from *products.csv*.  
- For each \(i\in I\):  
  - \(v_i\) = benefit coefficient (Value).  
  - \(w_i\) = development‑unit weight (Weight).  
- \(C\) = overall capacity (Capacity).  

Decision variables  

- \(x_i \in \mathbb{Z}_{\ge 0}\) – daily scale of development in area \(i\).

**Objective**

\[
\max \;\; \sum_{i\in I} v_i\,x_i
\]

**Capacity constraint**

\[
\sum_{i\in I} w_i\,x_i \;\le\; C
\]

**Variable domain**

\[
x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
\]

---

**Data Mapping**

| Parameter / Variable | Source Table | Column |
|----------------------|--------------|--------|
| \(I\) (index set)    | file_1_view_0 | ProductName |
| \(v_i\)              | file_1_view_0 | Value |
| \(w_i\)              | file_1_view_0 | Weight |
| \(C\)                | file_0_view_0 | Capacity |
| \(x_i\)              | – (decision variable) | – |

All tables and columns are referenced exactly as returned by CSVQA. No additional data or assumptions are introduced.