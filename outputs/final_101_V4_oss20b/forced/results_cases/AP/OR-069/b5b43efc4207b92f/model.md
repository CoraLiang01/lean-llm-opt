**Mathematical Model**

Let  

* \(M=\{m_1,\dots ,m_{11}\}\) be the set of managers (rows of the CSV).  
* \(P=\{p_1,\dots ,p_{11}\}\) be the set of projects (columns of the CSV).  

Parameters  

* \(c_{mp}\) – assignment cost of manager \(m\) to project \(p\).  

Decision variables  

* \(x_{mp}\in\{0,1\}\) – equals 1 if manager \(m\) is assigned to project \(p\), 0 otherwise.  

---

### Objective
\[
\min \;\sum_{m\in M}\sum_{p\in P} c_{mp}\,x_{mp}
\]

### Constraints
1. **Each manager is assigned to exactly one project**  
   \[
   \sum_{p\in P} x_{mp}=1 \quad \forall m\in M
   \]

2. **Each project is assigned to exactly one manager**  
   \[
   \sum_{m\in M} x_{mp}=1 \quad \forall p\in P
   \]

3. **Binary variables**  
   \[
   x_{mp}\in\{0,1\}\quad \forall m\in M,\;p\in P
   \]

---

**Data Mapping**

```json
{
  "table_id": "file_0_view_0",
  "columns": {
    "Manager": "M",
    "Project 1 Cost": "P1",
    "Project 2 Cost": "P2",
    "Project 3 Cost": "P3",
    "Project 4 Cost": "P4",
    "Project 5 Cost": "P5",
    "Project 6 Cost": "P6",
    "Project 7 Cost": "P7",
    "Project 8 Cost": "P8",
    "Project 9 Cost": "P9",
    "Project 10 Cost": "P10",
    "Project 11 Cost": "P11"
  },
  "parameter_mapping": {
    "c": {
      "row_index": "Manager",
      "column_index": "Project {p} Cost",
      "value": "numeric cost"
    }
  }
}
```

* `c[m,p]` is obtained from the cell in the row where `Manager = m` and the column named `"Project {p} Cost"`.  
* The index sets `M` and `P` are the distinct values appearing in the `Manager` column and the project‑cost columns, respectively.