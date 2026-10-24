**Mathematical Model**

Let  

* \(P\) – set of platforms (indexed by the values in *PlatformId*).  
* \(G\) – set of game genres (indexed by the values in *ProductName*).  

Parameters  

* \(C_p\) – memory capacity of platform \(p\).  
* \(V_g\) – predefined value of a unit of genre \(g\).  
* \(W_g\) – memory weight of a unit of genre \(g\).  

Decision variables  

* \(x_{p,g}\in \mathbb{Z}_{\ge 0}\) – number of units of genre \(g\) listed on platform \(p\).

\[
\begin{aligned}
\max\;& \sum_{p\in P}\sum_{g\in G} V_g\,x_{p,g} \\[4pt]
\text{s.t.}\;& \sum_{g\in G} W_g\,x_{p,g}\;\le\; C_p \qquad \forall p\in P \\[4pt]
& x_{p,g}\;\in\;\mathbb{Z}_{\ge 0}\qquad \forall p\in P,\;g\in G
\end{aligned}
\]

---

**Data Mapping**

```json
{
  "parameters": {
    "C_p": {
      "source_table": "file_0_view_0",
      "columns": ["PlatformId", "Capacity"]
    },
    "V_g": {
      "source_table": "file_1_view_0",
      "columns": ["ProductName", "Value"]
    },
    "W_g": {
      "source_table": "file_1_view_0",
      "columns": ["ProductName", "Weight"]
    }
  },
  "index_sets": {
    "P": {
      "source_table": "file_0_view_0",
      "column": "PlatformId"
    },
    "G": {
      "source_table": "file_1_view_0",
      "column": "ProductName"
    }
  }
}
```

This mapping links each symbolic parameter and index set directly to the corresponding columns in the provided CSV files. No numerical values are enumerated in the model.