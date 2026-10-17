**Mathematical Model**

Let  

- \(I=\{i\mid i\text{ is a ProductName in } \text{products.csv}\}\)  
- \(b_i\) – benefit coefficient for area \(i\)  
- \(C\) – overall development capacity  

Decision variables  

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

Objective  

\[
\max \;\sum_{i\in I} b_i\,x_i
\]

Capacity constraint  

\[
\sum_{i\in I} x_i \;\le\; C
\]

---

**Data Mapping**

| Parameter | Source Table | Source Columns | Mapping Rule |
|-----------|--------------|----------------|--------------|
| \(b_i\) | `file_1_view_0` | `ProductName`, `Value` | \(b_i = \text{Value}\) for the row where `ProductName = i` |
| \(C\) | `file_0_view_0` | `Capacity` | \(C = \text{Capacity}\) (single row) |

All other symbols are defined in the model above. No numerical values are enumerated.