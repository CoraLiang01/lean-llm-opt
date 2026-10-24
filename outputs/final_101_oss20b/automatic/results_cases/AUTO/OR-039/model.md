**Mathematical Model**

\[
\begin{aligned}
\max\;&\sum_{i\in I}\sum_{j\in J} \text{Value}_i\,x_{ij} \\[4pt]
\text{s.t.}\;&\sum_{i\in I}\text{Weight}_i\,x_{ij}\;\le\;\text{Capacity}_j,
\qquad \forall j\in J \\[4pt]
&x_{ij}\in\mathbb{Z}_{\ge0},
\qquad \forall i\in I,\;j\in J
\end{aligned}
\]

**Index Sets**

- \(I\): set of product types (ProductName)  
- \(J\): set of warehouses (Warehouse ID)

**Parameters**

- \(\text{Value}_i\): benefit coefficient of product \(i\)  
- \(\text{Weight}_i\): space (weight) requirement of product \(i\)  
- \(\text{Capacity}_j\): total capacity of warehouse \(j\)

**Variables**

- \(x_{ij}\): number of units of product \(i\) stored in warehouse \(j\) (integer, \(\ge0\))

---

### Data Mapping

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| \(\text{Capacity}_j\) | `file_0_view_0` | `Capacity` |
| \(J\) (Warehouse ID) | `file_0_view_0` | `Warehouse ID` |
| \(\text{Value}_i\) | `file_1_view_0` | `Value` |
| \(\text{Weight}_i\) | `file_1_view_0` | `Weight` |
| \(I\) (ProductName) | `file_1_view_0` | `ProductName` |

All parameters are taken directly from the corresponding CSV tables; no numerical values are enumerated in the model.