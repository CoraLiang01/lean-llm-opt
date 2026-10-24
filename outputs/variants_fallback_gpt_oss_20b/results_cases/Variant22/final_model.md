## Mathematical Model

**Index sets**  
\(C=\{C1,C2,C3,C4,C5,C6\}\) (families)  
\(O=\{O1,O2,O3\}\) (options per family)

**Parameters**  
\(v_{c,o}\) – value of option \(o\) in family \(c\) (from *option_catalog.csv*, column *Value*)  
\(w_{c,o}\) – weight of option \(o\) in family \(c\) (from *option_catalog.csv*, column *Weight*)  
\(l_{c,o}\) – labor‑hour requirement of option \(o\) in family \(c\) (from *option_catalog.csv*, column *LaborHours*)  
\(W\) – total weight limit (from *resource_limits.csv*, row where *Resource* = *Weight*)  
\(L\) – total labor‑hour limit (from *resource_limits.csv*, row where *Resource* = *LaborHours*)

**Decision variables**  
\(x_{c,o}\in\{0,1\}\) =1 if option \(o\) is selected from family \(c\), 0 otherwise.

**Objective**  
\[
\max \;\sum_{c\in C}\sum_{o\in O} v_{c,o}\,x_{c,o}
\]

**Constraints**

1. **Exactly one option per family**  
\[
\sum_{o\in O} x_{c,o}=1 \qquad \forall\,c\in C
\]

2. **Weight capacity**  
\[
\sum_{c\in C}\sum_{o\in O} w_{c,o}\,x_{c,o}\;\le\; W
\]

3. **Labor‑hour capacity**  
\[
\sum_{c\in C}\sum_{o\in O} l_{c,o}\,x_{c,o}\;\le\; L
\]

4. **Binary variables**  
\[
x_{c,o}\in\{0,1\}\qquad \forall\,c\in C,\;o\in O
\]

---

## Data Mapping

| Parameter | Source Table | Source Columns | Mapping Rule |
|-----------|--------------|----------------|--------------|
| \(v_{c,o}\) | `file_0_view_0` | `Value` | value of option \(o\) in family \(c\) |
| \(w_{c,o}\) | `file_0_view_0` | `Weight` | weight of option \(o\) in family \(c\) |
| \(l_{c,o}\) | `file_0_view_0` | `LaborHours` | labor‑hour requirement of option \(o\) in family \(c\) |
| \(W\) | `file_1_view_0` | `Limit` where `Resource`=`Weight` | total weight limit |
| \(L\) | `file_1_view_0` | `Limit` where `Resource`=`LaborHours` | total labor‑hour limit |

All index sets \(C\) and \(O\) are taken from the distinct values present in the *option_catalog.csv* table. No additional entities are introduced. The model uses only the data explicitly supplied in the observation.