**Mathematical Model**

Let  

- \(I\) be the set of storage areas (indexed by `StorageID`).  
- \(J\) be the set of air‑conditioner types (indexed by `ProductName`).  

Parameters  

- \(c_i\) : capacity of storage area \(i\)  (from column **Capacity** in table `file_0_view_0`).  
- \(v_j\) : value of product \(j\)  (from column **Value** in table `file_1_view_0`).  
- \(w_j\) : weight (size) of product \(j\)  (from column **Weight** in table `file_1_view_0`).  

Decision variables  

- \(x_{ij}\in\mathbb{Z}_{\ge 0}\) : number of units of product \(j\) placed in storage area \(i\).

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{j\in J} w_j\,x_{ij} \;\le\; c_i \qquad \forall i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0}\qquad \forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

```json
{
  "parameters": {
    "c_i": {
      "source_table": "file_0_view_0",
      "source_column": "Capacity",
      "index_column": "StorageID"
    },
    "v_j": {
      "source_table": "file_1_view_0",
      "source_column": "Value",
      "index_column": "ProductName"
    },
    "w_j": {
      "source_table": "file_1_view_0",
      "source_column": "Weight",
      "index_column": "ProductName"
    }
  },
  "variables": {
    "x_ij": {
      "type": "integer",
      "lower_bound": 0
    }
  }
}
```

This model uses all entities returned by the current observation and respects the integer requirement for the decision variables.