**Mathematical Model**

Let  

- \(I\) be the set of shelves indexed by `ShelfID` from **file_0_view_0**.  
- \(J\) be the set of products indexed by `ProductName` from **file_1_view_0**.  

Parameters  

\[
\begin{aligned}
&\text{capacity}[i] &&\text{(capacity of shelf }i\text{)} \\
&\text{value}[j] &&\text{(value of product }j\text{)} \\
&\text{weight}[j] &&\text{(weight of product }j\text{)}
\end{aligned}
\]

Decision variables  

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\quad \forall i\in I,\; j\in J
\]

Objective  

\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{value}[j]\;x_{ij}
\]

Constraints  

\[
\sum_{j\in J}\text{weight}[j]\;x_{ij}\;\le\;\text{capacity}[i]\quad \forall i\in I
\]

---

**Data Mapping**

```json
{
  "capacity": {
    "source_table": "file_0_view_0",
    "index_column": "ShelfID",
    "value_column": "Capacity"
  },
  "value": {
    "source_table": "file_1_view_0",
    "index_column": "ProductName",
    "value_column": "Value"
  },
  "weight": {
    "source_table": "file_1_view_0",
    "index_column": "ProductName",
    "value_column": "Weight"
  }
}
```