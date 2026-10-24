**Mathematical Model**

Let  

* \(P=\{I,II,III\}\) – products  
* \(E_A=\{A1,A2,A3\}\) – equipment that performs procedure A  
* \(E_B=\{B1,B2,B3\}\) – equipment that performs procedure B  

For every \((e,p)\in E_A\times P\) let  

\[
t_{e,p}\;(\text{h/unit})\quad\text{processing time of product }p\text{ on equipment }e
\]

and for every \((e,p)\in E_B\times P\) let the same symbol denote the processing time of procedure B.  
If a value is missing in the data, \(t_{e,p}=0\) and the corresponding production is forbidden.

Let  

* \(Q_p\ge 0\) – total quantity of product \(p\) produced  
* \(x_{e,p}\ge 0\) – quantity of product \(p\) processed on equipment \(e\in E_A\)  
* \(y_{e,p}\ge 0\) – quantity of product \(p\) processed on equipment \(e\in E_B\)  

Parameters  

* \(a_e\) – available operating time of equipment \(e\) (h)  
* \(c_e\) – equipment cost per hour of equipment \(e\) (yuan/h)  
* \(r_p\) – raw‑material cost per unit of product \(p\) (yuan)  
* \(s_p\) – selling price per unit of product \(p\) (yuan)

The model is

\[
\begin{aligned}
\max\;&
\sum_{p\in P} s_p\,Q_p
-\sum_{p\in P} r_p\,Q_p
-\sum_{e\in E_A}\;c_e\!\!\sum_{p\in P} t_{e,p}\,x_{e,p}
-\sum_{e\in E_B}\;c_e\!\!\sum_{p\in P} t_{e,p}\,y_{e,p}
\\[4pt]
\text{s.t.}\;&
\sum_{e\in E_A} x_{e,p}=Q_p \qquad\forall p\in P \quad\text{(procedure A)}\\
&\sum_{e\in E_B} y_{e,p}=Q_p \qquad\forall p\in P \quad\text{(procedure B)}\\[4pt]
&\sum_{p\in P} t_{e,p}\,x_{e,p}\le a_e \qquad\forall e\in E_A \quad\text{(time on A)}\\
&\sum_{p\in P} t_{e,p}\,y_{e,p}\le a_e \qquad\forall e\in E_B \quad\text{(time on B)}\\[4pt]
&x_{e,p}=0 \;\text{if}\; t_{e,p}=0 \qquad\forall e\in E_A,\;p\in P\\
&y_{e,p}=0 \;\text{if}\; t_{e,p}=0 \qquad\forall e\in E_B,\;p\in P\\[4pt]
&Q_p\ge0,\;x_{e,p}\ge0,\;y_{e,p}\ge0 \qquad\forall p,e
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column(s) | Notes |
|-----------|--------|----------|-----------|-------|
| \(t_{e,p}\) | Processing time | `file_0_view_0` | Row where `Equipment / Cost` = *e*, column for product *p* | Blank → 0 (forbidden) |
| \(a_e\) | Available operating time | `file_0_view_0` | Row where `Equipment / Cost` = *e*, column `Available Equipment Operating Time` | |
| \(c_e\) | Equipment cost per hour | `file_0_view_0` | Row where `Equipment / Cost` = *e*, column `Equipment Cost at Full Load (yuan)` | |
| \(r_p\) | Raw‑material cost | `file_0_view_0` | Row where `Equipment / Cost` = `Raw Material Cost (yuan/unit)`, column for product *p* | |
| \(s_p\) | Unit selling price | `file_0_view_0` | Row where `Equipment / Cost` = `Unit Price (yuan/unit)`, column for product *p* | |

*Index sets*  
- \(E_A=\{A1,A2,A3\}\) (rows with `Equipment / Cost` starting with “A”)  
- \(E_B=\{B1,B2,B3\}\) (rows with `Equipment / Cost` starting with “B”)  
- \(P=\{I,II,III\}\) (products identified by column names `Product I`, `Product II`, `Product III`)

All other rows (e.g., `B4`) are ignored because they are not part of the equipment sets defined in the problem statement.