**Mathematical Model**

Indices  
\(p \in \{I,\,II,\,III\}\) products  
\(e \in \{A1,\,A2,\,A3,\,B1,\,B2,\,B3,\,B4\}\) equipment  
\(A=\{A1,\,A2,\,A3\}\) procedure A equipment  
\(B=\{B1,\,B2,\,B3,\,B4\}\) procedure B equipment  

Parameters  

\[
\begin{aligned}
t_{e,p} &\;=\;\text{processing time per unit of product }p\text{ on equipment }e\\
a_{e}   &\;=\;\text{available operating time of equipment }e\\
c_{e}   &\;=\;\text{equipment cost per hour of equipment }e\\
r_{p}   &\;=\;\text{raw‑material cost per unit of product }p\\
s_{p}   &\;=\;\text{unit selling price of product }p
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
x_{p} &\ge 0 &&\text{units of product }p\text{ produced}\\
z_{e,p}&\ge 0 &&\text{units of product }p\text{ processed on equipment }e
\end{aligned}
\]

Allowed processing combinations  

\[
\begin{aligned}
(e,p)\in\mathcal{A}\iff
\begin{cases}
p=I &\Rightarrow e\in\{A1,A2,A3,B1,B2,B3,B4\}\\
p=II&\Rightarrow e\in\{A1,A2,A3,B1\}\\
p=III&\Rightarrow e\in\{A2,B2\}
\end{cases}
\end{aligned}
\]

(Variables \(z_{e,p}\) are defined only for \((e,p)\in\mathcal{A}\); otherwise they are zero.)

Objective  

\[
\max \;\sum_{p\in\{I,II,III\}}\bigl(s_{p}-r_{p}\bigr)x_{p}
-\sum_{(e,p)\in\mathcal{A}}c_{e}\,t_{e,p}\,z_{e,p}
\]

Constraints  

1. **Equipment capacity** for every \(e\in E\)

\[
\sum_{p\in\{I,II,III\}} t_{e,p}\,z_{e,p}\;\le\; a_{e}
\]

2. **Procedure A allocation** for every \(p\in\{I,II,III\}\)

\[
\sum_{e\in A} z_{e,p}\;=\;x_{p}
\]

3. **Procedure B allocation** for every \(p\in\{I,II,III\}\)

\[
\sum_{e\in B} z_{e,p}\;=\;x_{p}
\]

4. **Non‑negativity**

\[
x_{p}\ge0,\qquad z_{e,p}\ge0\;\;\forall (e,p)\in\mathcal{A}
\]

---

**Data Mapping**

| Parameter | Source Table | Row Identifier | Column(s) |
|-----------|--------------|----------------|-----------|
| \(t_{e,p}\) | file_0_view_0 | `"Equipment / Cost" = e` | `"Product I"`, `"Product II"`, `"Product III"` |
| \(a_{e}\) | file_0_view_0 | `"Equipment / Cost" = e` | `"Available Equipment Operating Time"` |
| \(c_{e}\) | file_0_view_0 | `"Equipment / Cost" = e` | `"Equipment Cost at Full Load (yuan)"` |
| \(r_{p}\) | file_0_view_0 | `"Equipment / Cost" = "Raw Material Cost (yuan/unit)"` | `"Product I"`, `"Product II"`, `"Product III"` |
| \(s_{p}\) | file_0_view_0 | `"Equipment / Cost" = "Unit Price (yuan/unit)"` | `"Product I"`, `"Product II"`, `"Product III"` |

*Missing \(t_{e,p}\) entries in the table imply that the product cannot be processed on that equipment (i.e., \((e,p)\notin\mathcal{A}\)).*