**Mathematical Model**

Let  

\[
S=\{s\mid s \text{ is a Walmart store}\},\qquad  
C=\{c\mid c \text{ is a customer group}\}.
\]

Parameters  

\[
\begin{aligned}
d_c &\quad\text{demand of customer }c,\\
s_s &\quad\text{daily supply capacity of store }s,\\
c_{sc}&\quad\text{transportation cost per unit from store }s\text{ to customer }c.
\end{aligned}
\]

Decision variables  

\[
x_{sc}\in \mathbb{Z}_{\ge 0}\quad\text{units shipped from }s\text{ to }c .
\]

\[
\begin{aligned}
\min\;&\sum_{s\in S}\sum_{c\in C} c_{sc}\,x_{sc} \\[4pt]
\text{s.t. }&\sum_{s\in S} x_{sc}= d_c , &&\forall c\in C \quad\text{(demand satisfaction)}\\[4pt]
&\sum_{c\in C} x_{sc}\le s_s , &&\forall s\in S \quad\text{(supply capacity)}\\[4pt]
&x_{sc}\in \mathbb{Z}_{\ge 0}, &&\forall s\in S,\;c\in C .
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(d_c\) | `file_0_view_0` | `demand` |
| \(s_s\) | `file_1_view_0` | `supply_capacity` |
| \(c_{sc}\) | `file_2_view_0` | `C1`–`C12` (columns named after customers) |

| Index Set | Source Table | Identifier Column |
|-----------|--------------|-------------------|
| \(S\) | `file_1_view_0` | `Unnamed: 0` |
| \(C\) | `file_0_view_0` | `customer` |

The cost matrix `file_2_view_0` is aligned such that row identifier `Unnamed: 0` matches \(S\) and column identifiers `C1`–`C12` match \(C\). No other data are required.