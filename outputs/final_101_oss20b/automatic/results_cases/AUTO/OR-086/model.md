## Mathematical Model

**Sets**  
\(G=\{I,II,III\}\) grades of raw wine  
\(B=\{Red,Yellow,Blue\}\) wine brands  

**Parameters**  

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(s_g\) | Daily supply limit of grade \(g\) (kg) | `file_0_view_0` – “Daily Supply (kg)” |
| \(c_g\) | Unit cost of grade \(g\) (CNY/kg) | `file_0_view_0` – “Cost (CNY/kg)” |
| \(p_b\) | Selling price of brand \(b\) (CNY/kg) | `file_1_view_0` – “Selling Price (CNY/kg)” |
| \(\overline{u}_{b,g}\) | Upper‑bound proportion of grade \(g\) in brand \(b\) (fraction) | parsed from `file_1_view_0` – “Blending Requirements” |
| \(\underline{u}_{b,g}\) | Lower‑bound proportion of grade \(g\) in brand \(b\) (fraction) | parsed from `file_1_view_0` – “Blending Requirements” |

(If a brand has no stated bound for a grade, the corresponding \(\overline{u}_{b,g}=1\) and \(\underline{u}_{b,g}=0\); such bounds are omitted from the constraint list.)

**Decision Variables**  

\[
x_{g,b}\ge 0 \quad\text{(kg of grade }g\text{ used for brand }b)
\]
\[
P_b \ge 0 \quad\text{(total kg produced of brand }b)
\]

**Objective**  

\[
\max \;\sum_{b\in B} p_b\,P_b \;-\; \sum_{g\in G} c_g\sum_{b\in B} x_{g,b}
\]

**Constraints**

1. **Brand production definition**  
   \[
   P_b = \sum_{g\in G} x_{g,b}\qquad \forall\,b\in B
   \]

2. **Blending proportion bounds**  
   \[
   \underline{u}_{b,g}\,P_b \;\le\; x_{g,b}\;\le\;\overline{u}_{b,g}\,P_b
   \qquad \forall\,b\in B,\; g\in G \text{ with a stated bound}
   \]

3. **Raw‑material supply limits**  
   \[
   \sum_{b\in B} x_{g,b}\;\le\; s_g \qquad \forall\,g\in G
   \]

4. **Minimum production of Red**  
   \[
   P_{\text{Red}}\;\ge\;2000
   \]

5. **Non‑negativity**  
   \[
   x_{g,b}\ge0,\; P_b\ge0 \qquad \forall\,g\in G,\;b\in B
   \]

---

## Data Mapping

| Table ID | Column | Parameter | Mapping Rule |
|----------|--------|-----------|--------------|
| `file_0_view_0` | `Grade` | \(g\) | Index set \(G\) |
| `file_0_view_0` | `Daily Supply (kg)` | \(s_g\) | \(s_{\text{Grade}} = \text{value}\) |
| `file_0_view_0` | `Cost (CNY/kg)` | \(c_g\) | \(c_{\text{Grade}} = \text{value}\) |
| `file_1_view_0` | `Brand` | \(b\) | Index set \(B\) |
| `file_1_view_0` | `Selling Price (CNY/kg)` | \(p_b\) | \(p_{\text{Brand}} = \text{value}\) |
| `file_1_view_0` | `Blending Requirements` | \(\overline{u}_{b,g}, \underline{u}_{b,g}\) | Parse textual rule; e.g. “I less than 10%” → \(\overline{u}_{Red,I}=0.10\); “II more than 50%” → \(\underline{u}_{Red,II}=0.50\). |

No additional data or constraints are introduced beyond those explicitly stated in the query and the source tables.