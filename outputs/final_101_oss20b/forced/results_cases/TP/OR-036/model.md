**Mathematical Model**

Let  

* \(I\) – set of product names (vehicle types).  
* \(c_i\) – benefit coefficient (value) of product \(i\).  
* \(w_i\) – weight (inventory consumption) of product \(i\).  
* \(C\) – total inventory capacity.  

Decision variables  

\[
x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
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

**Data Mapping**

| Parameter | Source Table | Column | Mapping |
|-----------|--------------|--------|---------|
| \(C\) | `file_0_view_0` | `Capacity` | `C = Capacity` |
| \(c_i\) | `file_1_view_0` | `Value` | `c_i = Value` for each product |
| \(w_i\) | `file_1_view_0` | `Weight` | `w_i = Weight` for each product |
| \(I\) | `file_1_view_0` | `ProductName` | `I = {ProductName}` |

All parameters are taken directly from the corresponding CSV files; no additional data are introduced. The model uses integer decision variables as required.