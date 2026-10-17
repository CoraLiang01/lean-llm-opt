**Mathematical Model**

Let  

* \(I\) be the set of area names (rows of *products.csv*).  
* For each \(i\in I\):  
  * \(v_i\) – benefit coefficient (column **Value**).  
  * \(w_i\) – weight per unit of development (column **Weight**).  
* \(C\) – overall development capacity (column **Capacity** of *capacity.csv*).  

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

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(C\) | `file_0_view_0` | `Capacity` |
| \(I\) | `file_1_view_0` | `ProductName` |
| \(v_i\) | `file_1_view_0` | `Value` |
| \(w_i\) | `file_1_view_0` | `Weight` |

All indices, parameters, and variables are defined exactly as they appear in the source tables; no values are enumerated in the model.