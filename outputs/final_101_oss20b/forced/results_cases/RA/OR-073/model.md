**Mathematical model**

\[
\begin{aligned}
\text{Maximize}\quad
&\sum_{p\in P}\; \pi_p \;\sum_{e\in E}x_{p e}
 \;-\;\sum_{e\in E}\;c_e\;\sum_{p\in P}t_{p e}\,x_{p e} \\[4pt]
\text{subject to}\quad
&\sum_{p\in P}t_{p e}\,x_{p e}\;\le\;A_e
 \qquad\forall e\in E \\[4pt]
&x_{p e}\;\ge\;0
 \qquad\forall (p,e)\in\mathcal{A}
\end{aligned}
\]

*Indices*  
\(P=\{I,\,II,\,III\}\) products  
\(E=\{A1,\,A2,\,B1\}\) equipment types  
\(\mathcal{A}=\{(I,A1),(I,A2),(I,B1),\\
\quad\;(II,A1),(II,A2),(II,B1),\\
\quad\;(III,A2)\}\) allowed product–equipment pairs  

*Parameters*  
\(\pi_p\) profit per unit of product \(p\) (selling price – raw‑material cost)  
\(t_{p e}\) processing time per unit of product \(p\) on equipment \(e\)  
\(A_e\) available operating time of equipment \(e\)  
\(c_e\) equipment cost per unit time of equipment \(e\)  

*Decision variables*  
\(x_{p e}\) continuous, number of units of product \(p\) processed on equipment \(e\)  

---

### Data mapping (all parameters are taken from the CSV file `43.csv`)

| Parameter | Table ID | Column | Row filter (Equipment / Cost) |
|-----------|----------|--------|------------------------------|
| \(\pi_p\) | – | – | – |
| \(t_{I,A1}\) | file_0_view_0 | Product I | A1 |
| \(t_{I,A2}\) | file_0_view_0 | Product I | A2 |
| \(t_{I,B1}\) | file_0_view_0 | Product I | B1 |
| \(t_{II,A1}\) | file_0_view_0 | Product II | A1 |
| \(t_{II,A2}\) | file_0_view_0 | Product II | A2 |
| \(t_{II,B1}\) | file_0_view_0 | Product II | B1 |
| \(t_{III,A2}\) | file_0_view_0 | Product III | A2 |
| \(A_{A1}\) | file_0_view_0 | Available Equipment Operating Time | A1 |
| \(A_{A2}\) | file_0_view_0 | Available Equipment Operating Time | A2 |
| \(A_{B1}\) | file_0_view_0 | Available Equipment Operating Time | B1 |
| \(c_{A1}\) | file_0_view_0 | Equipment Cost at Full Load (yuan) | A1 |
| \(c_{A2}\) | file_0_view_0 | Equipment Cost at Full Load (yuan) | A2 |
| \(c_{B1}\) | file_0_view_0 | Equipment Cost at Full Load (yuan) | B1 |

*Note:* The profit per unit \(\pi_p\) is not supplied in the data; it is treated as a free parameter to be supplied by the user. All other parameters are extracted directly from the rows of `43.csv` as shown above.