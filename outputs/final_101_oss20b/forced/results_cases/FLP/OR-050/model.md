**Mathematical Model**

Let  

* \(I=\{\,i\mid \text{ShelfID}=i\text{ in } \text{file\_0\_view\_0}\,\}\)  
* \(J=\{\,j\mid \text{ProductName}=j\text{ in } \text{file\_1\_view\_0}\,\}\)  
* \(j^{*}\) be the first product in the order of appearance in *products.csv* (i.e. “Smartphone”).

Decision variables  
\[
x_{ij}\in \mathbb{Z}_{+}\qquad \forall i\in I,\;j\in J
\]

Parameters  
\[
\begin{aligned}
\text{capacity}_i &\;=\;\text{Capacity of shelf }i \quad (\text{file\_0\_view\_0})\\
\text{value}_j &\;=\;\text{Value of product }j \quad (\text{file\_1\_view\_0})\\
\text{weight}_j &\;=\;\text{Weight of product }j \quad (\text{file\_1\_view\_0})
\end{aligned}
\]

Objective  
\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{value}_j\,x_{ij}
\]

Constraints  

1. **Capacity of each shelf**  
\[
\sum_{j\in J}\text{weight}_j\,x_{ij}\;\le\;\text{capacity}_i
\qquad \forall i\in I
\]

2. **Minimum quantity of the first product**  
\[
\sum_{i\in I}x_{i\,j^{*}}\;\ge\;5
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(I\) | `file_0_view_0` | `ShelfID` |
| \(\text{capacity}_i\) | `file_0_view_0` | `Capacity` |
| \(J\) | `file_1_view_0` | `ProductName` |
| \(\text{value}_j\) | `file_1_view_0` | `Value` |
| \(\text{weight}_j\) | `file_1_view_0` | `Weight` |
| \(j^{*}\) | `file_1_view_0` | first row’s `ProductName` (“Smartphone”) |

All other indices and parameters are derived from these mappings. No numerical values are enumerated in the model.