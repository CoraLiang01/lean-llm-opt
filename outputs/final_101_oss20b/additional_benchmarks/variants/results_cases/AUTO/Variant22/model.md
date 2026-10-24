## Mathematical Model

**Index sets**  
\(C=\{C1,C2,C3,C4,C5,C6\}\) (families)  
\(O=\{O1,O2,O3\}\) (options per family)

**Parameters**  
\(v_{c,o}\) – value of option \(o\) in family \(c\)  
\(w_{c,o}\) – weight of option \(o\) in family \(c\)  
\(l_{c,o}\) – labor‑hour requirement of option \(o\) in family \(c\)  
\(W_{\max}\) – maximum total weight (55)  
\(L_{\max}\) – maximum total labor hours (64)

**Decision variables**  
\(x_{c,o}\in\{0,1\}\) (1 if option \(o\) is chosen from family \(c\), 0 otherwise)

**Objective**  
\[
\max \;\sum_{c\in C}\sum_{o\in O} v_{c,o}\,x_{c,o}
\]

**Constraints**

1. **Exactly one option per family**  
\[
\sum_{o\in O} x_{c,o}=1 \qquad \forall\,c\in C
\]

2. **Weight limit**  
\[
\sum_{c\in C}\sum_{o\in O} w_{c,o}\,x_{c,o}\;\le\; W_{\max}
\]

3. **Labor‑hour limit**  
\[
\sum_{c\in C}\sum_{o\in O} l_{c,o}\,x_{c,o}\;\le\; L_{\max}
\]

4. **Binary variables**  
\[
x_{c,o}\in\{0,1\}\qquad \forall\,c\in C,\;o\in O
\]

---

## Data Mapping

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(v_{c,o}\) | option_catalog.csv | file_0_view_0 | Value |
| \(w_{c,o}\) | option_catalog.csv | file_0_view_0 | Weight |
| \(l_{c,o}\) | option_catalog.csv | file_0_view_0 | LaborHours |
| \(W_{\max}\) | resource_limits.csv | file_1_view_0 | Limit (where Resource = Weight) |
| \(L_{\max}\) | resource_limits.csv | file_1_view_0 | Limit (where Resource = LaborHours) |

*Index sets \(C\) and \(O\) are derived from the distinct values of the **Family** and **Option** columns in `option_catalog.csv`.*