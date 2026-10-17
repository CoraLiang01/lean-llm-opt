**Mathematical Model**

Let  

* \(I=\{i\mid i\text{ is a ShelfID in }{\tt capacity.csv}\}\)  
* \(J=\{j\mid j\text{ is a ProductName in }{\tt products.csv}\}\)

Parameters  

* \(C_i\) – capacity of shelf \(i\)  
* \(v_j\) – value of one unit of product \(j\)  
* \(w_j\) – weight of one unit of product \(j\)

Decision variables  

\[
x_{ij}\in \mathbb{Z}_{+}\qquad \forall i\in I,\; j\in J
\]

---

**Objective**

\[
\max \;\sum_{i\in I}\sum_{j\in J} v_j\,x_{ij}
\]

---

**Constraints**

1. **Shelf capacity**

\[
\sum_{j\in J} w_j\,x_{ij}\;\le\; C_i \qquad \forall i\in I
\]

2. **Non‑negativity & integrality**

\[
x_{ij}\in \mathbb{Z}_{+}\qquad \forall i\in I,\; j\in J
\]

---

**Data Mapping**

```json
{
  "parameters": {
    "C_i": {
      "source_table": "file_0_view_0",
      "key_column": "ShelfID",
      "value_column": "Capacity"
    },
    "v_j": {
      "source_table": "file_1_view_0",
      "key_column": "ProductName",
      "value_column": "Value"
    },
    "w_j": {
      "source_table": "file_1_view_0",
      "key_column": "ProductName",
      "value_column": "Weight"
    }
  },
  "index_sets": {
    "I": "file_0_view_0.ShelfID",
    "J": "file_1_view_0.ProductName"
  }
}
```

This model maximizes the total value of products placed on shelves while respecting each shelf’s weight capacity, with integer quantities of each product.