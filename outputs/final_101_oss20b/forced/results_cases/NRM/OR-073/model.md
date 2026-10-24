**Symbolic Model**

```markdown
**Index sets**
- \(P = \{I,\,II,\,III\}\)  (products)  
- \(A = \{A1,\,A2,\,A3\}\)  (equipment for procedure A)  
- \(B = \{B1,\,B2,\,B3,\,B4\}\) (equipment for procedure B)  
- \(E = A \cup B\)  (all equipment)

**Parameters**
- \(t_{p,e}\) processing time per unit of product \(p\) on equipment \(e\)  
- \(T_e\) available operating time of equipment \(e\)  
- \(c_e\) equipment cost per hour of equipment \(e\)  
- \(r_p\) raw‑material cost per unit of product \(p\)  
- \(p_p\) unit selling price of product \(p\)  
- \(\text{allow}_{p,e}\in\{0,1\}\) 1 if product \(p\) may be processed on equipment \(e\), 0 otherwise  

**Decision variables**
- \(y_{p,e}\ge 0\) number of units of product \(p\) processed on equipment \(e\)  
  (defined only for \(\text{allow}_{p,e}=1\))

**Objective**
\[
\max \;\sum_{p\in P}\;\sum_{e\in E}\;
y_{p,e}\;\Bigl(p_p-r_p-c_e\,t_{p,e}\Bigr)
\]

**Constraints**

1. **Equipment‑time limits**  
   \[
   \sum_{p\in P} y_{p,e}\,t_{p,e}\;\le\;T_e,
   \qquad \forall\,e\in E
   \]

2. **Procedure consistency**  
   \[
   \sum_{e\in A} y_{p,e}\;=\;\sum_{e\in B} y_{p,e},
   \qquad \forall\,p\in P
   \]

3. **Non‑negativity**  
   \[
   y_{p,e}\;\ge\;0,
   \qquad \forall\,p\in P,\;e\in E
   \]

*(The variables are continuous by definition.)*


---

**Data Mapping**

| Parameter | Source table | Source column | Notes |
|-----------|--------------|---------------|-------|
| \(t_{p,e}\) | `file_0_view_0` | Column for product \(p\) (e.g., “Product I”) | Row where “Equipment / Cost” = \(e\) |
| \(T_e\) | `file_0_view_0` | “Available Equipment Operating Time” | Row where “Equipment / Cost” = \(e\) |
| \(c_e\) | `file_0_view_0` | “Equipment Cost at Full Load (yuan)” | Row where “Equipment / Cost” = \(e\) |
| \(r_p\) | `file_0_view_0` | Column for product \(p\) | Row where “Equipment / Cost” = “Raw Material Cost (yuan/unit)” |
| \(p_p\) | `file_0_view_0` | Column for product \(p\) | Row where “Equipment / Cost” = “Unit Price (yuan/unit)” |
| \(\text{allow}_{p,e}\) | *Derived* | – | Defined by problem description:  
 • Product I: all \(e\in A\cup B\)  
 • Product II: all \(e\in A\) and \(e=B1\) only  
 • Product III: \(e=A2\) and \(e=B2\) only |

*All other data (e.g., numeric values) are taken directly from the specified table and columns; no additional values are introduced.*