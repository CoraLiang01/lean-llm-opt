**Mathematical Model**

Let  

- \(P\) be the set of platforms indexed by the values in column **PlatformId** of *capacity.csv* (table_id = `file_0_view_0`).  
- \(G\) be the set of game genres indexed by the values in column **ProductName** of *products.csv* (table_id = `file_1_view_0`).  

Parameters  

- \(\text{Cap}[p]\) = capacity of platform \(p\) (units of memory).  
- \(\text{Val}[g]\) = value of one unit of genre \(g\).  
- \(\text{Wgt}[g]\) = memory weight of one unit of genre \(g\).  

Decision variables  

- \(x_{p,g}\) = number of units of genre \(g\) to list on platform \(p\).  
  \(x_{p,g}\in\mathbb{Z}_{\ge 0}\).

Objective  

\[
\max \;\sum_{p\in P}\sum_{g\in G}\text{Val}[g]\;x_{p,g}
\]

Capacity constraints (one per platform)

\[
\sum_{g\in G}\text{Wgt}[g]\;x_{p,g}\;\le\;\text{Cap}[p],
\qquad \forall\,p\in P
\]

All other constraints are implicit in the variable domains.

---

**Data Mapping**

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| \(\text{Cap}[p]\) | `file_0_view_0` | `Capacity` |
| \(\text{Val}[g]\) | `file_1_view_0` | `Value` |
| \(\text{Wgt}[g]\) | `file_1_view_0` | `Weight` |

Indices are taken directly from the distinct values of the corresponding columns in the supplied tables. No additional data or constraints are introduced beyond those required by the query.