**Mathematical Model**

Let  

* \(G=\{I,II,III\}\) grades of raw wine  
* \(B=\{Red,Yellow,Blue\}\) wine brands  

Decision variables  
\[
x_{b,g}\;\ge 0 \qquad\text{(kg of grade \(g\) used in brand \(b\))}
\]
Define the daily production of brand \(b\) as  
\[
P_b=\sum_{g\in G}x_{b,g}\; .
\]

Parameters  

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(s_g\) | Daily supply limit of grade \(g\) | `file_0_view_0` – column **Daily Supply (kg)** |
| \(c_g\) | Unit cost of grade \(g\) | `file_0_view_0` – column **Cost (CNY/kg)** |
| \(p_b\) | Selling price of brand \(b\) | `file_1_view_0` – column **Selling Price (CNY/kg)** |
| \(\underline{l}_{b,g}\) | Lower‑bound proportion of grade \(g\) in brand \(b\) (0 if not specified) | `file_1_view_0` – column **Blending Requirements** (parse “more than X%”) |
| \(\overline{u}_{b,g}\) | Upper‑bound proportion of grade \(g\) in brand \(b\) (1 if not specified) | `file_1_view_0` – column **Blending Requirements** (parse “less than X%”) |
| \(P^{\min}_{Red}\) | Minimum production of Red brand | constant 2000 kg |

---

### Objective
\[
\max \;\; \sum_{b\in B} p_b\,P_b \;-\; \sum_{b\in B}\sum_{g\in G} c_g\,x_{b,g}
\]

---

### Constraints  

1. **Blending lower bounds** (for every \((b,g)\) with \(\underline{l}_{b,g}>0\))  
   \[
   x_{b,g}\;\ge\;\underline{l}_{b,g}\,P_b \qquad \forall\,b\in B,\;g\in G
   \]

2. **Blending upper bounds** (for every \((b,g)\) with \(\overline{u}_{b,g}<1\))  
   \[
   x_{b,g}\;\le\;\overline{u}_{b,g}\,P_b \qquad \forall\,b\in B,\;g\in G
   \]

3. **Raw‑material supply limits**  
   \[
   \sum_{b\in B} x_{b,g}\;\le\; s_g \qquad \forall\,g\in G
   \]

4. **Minimum production of Red**  
   \[
   P_{Red}\;\ge\; P^{\min}_{Red}
   \]

5. **Non‑negativity**  
   \[
   x_{b,g}\;\ge 0 \qquad \forall\,b\in B,\;g\in G
   \]

---

**Data Mapping Rules**

| Parameter | Table ID | Column | Transformation |
|-----------|----------|--------|----------------|
| \(s_g\) | `file_0_view_0` | `Daily Supply (kg)` | numeric |
| \(c_g\) | `file_0_view_0` | `Cost (CNY/kg)` | numeric |
| \(p_b\) | `file_1_view_0` | `Selling Price (CNY/kg)` | numeric |
| \(\underline{l}_{b,g}\) | `file_1_view_0` | `Blending Requirements` | parse each record: for every phrase “\(g\) more than X%” set \(\underline{l}_{b,g}=X/100\); otherwise 0 |
| \(\overline{u}_{b,g}\) | `file_1_view_0` | `Blending Requirements` | parse each record: for every phrase “\(g\) less than X%” set \(\overline{u}_{b,g}=X/100\); otherwise 1 |
| \(P^{\min}_{Red}\) | constant | – | 2000 |

All index sets \(G\) and \(B\) are taken from the distinct values of the **Grade** and **Brand** columns, respectively. No other data are required.