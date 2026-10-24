##### Model

**Index set**  
\(I=\{i \mid i \text{ is a Project ID in } \texttt{project.csv}\}\)

**Parameters**  
\(c_i \;(\text{Capital in k$})\)  \(c_i = \texttt{project.csv}[i]["Capital (k$)"]\)  
\(npv_i \;(\text{NPV in k$})\)  \(npv_i = \texttt{project.csv}[i]["NPV (k$)"]\)

**Decision variables**  
\(x_i \in \{0,1\}\)  \(x_i=1\) if project \(i\) is selected, 0 otherwise.

**Objective**  
\[
\max \;\sum_{i\in I} npv_i\,x_i
\]

**Constraints**

1. **Budget**  
\[
\sum_{i\in I} c_i\,x_i \;\le\; 1000
\]

2. **Mutually exclusive (Projects 4 & 7)**  
\[
x_4 + x_7 \;\le\; 1
\]

3. **Prerequisite (Project 6 requires 1)**  
\[
x_6 \;\le\; x_1
\]

4. **Contingent (Project 10 requires 5)**  
\[
x_{10} \;\le\; x_5
\]

---

##### Data Mapping

| Parameter | Source | Mapping |
|-----------|--------|---------|
| \(c_i\) | `file_0_view_0` | `project.csv[Project ID=i]["Capital (k$)"]` |
| \(npv_i\) | `file_0_view_0` | `project.csv[Project ID=i]["NPV (k$)"]` |
| \(x_i\) | Decision variable | Binary variable for each Project ID |

*Table ID:* `file_0_view_0`  
*Columns used:* `"Project ID"`, `"Capital (k$)"`, `"NPV (k$)"`