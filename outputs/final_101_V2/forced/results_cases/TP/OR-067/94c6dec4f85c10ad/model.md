##### Decision Variables

Let $x_{ij} \in \{0,1\}$ for all managers $i \in I$ and projects $j \in J$:
- $x_{ij} = 1$ if manager $i$ is assigned to project $j$, $0$ otherwise.

##### Parameters

- $I = \{\text{MA}, \text{MB}, \text{MC}\}$ (managers, in source order)
- $J = \{\text{P1}, \text{P2}, \text{P3}\}$ (projects, in source order)
- Cost coefficients $c_{ij}$:

\[
\begin{array}{c|ccc}
      & \text{P1} & \text{P2} & \text{P3} \\
\hline
\text{MA} & 3000 & 3200 & 3100 \\
\text{MB} & 2800 & 3300 & 2900 \\
\text{MC} & 2900 & 3100 & 3000 \\
\end{array}
\]

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]
That is,
\[
\min \left(
3000\,x_{\text{MA},\text{P1}} + 3200\,x_{\text{MA},\text{P2}} + 3100\,x_{\text{MA},\text{P3}}
+ 2800\,x_{\text{MB},\text{P1}} + 3300\,x_{\text{MB},\text{P2}} + 2900\,x_{\text{MB},\text{P3}}
+ 2900\,x_{\text{MC},\text{P1}} + 3100\,x_{\text{MC},\text{P2}} + 3000\,x_{\text{MC},\text{P3}}
\right)
\]

##### Constraints

1. Each manager is assigned to exactly one project:
   \[
   \sum_{j \in J} x_{ij} = 1 \quad \forall i \in I
   \]
   Explicitly:
   \begin{align*}
   x_{\text{MA},\text{P1}} + x_{\text{MA},\text{P2}} + x_{\text{MA},\text{P3}} &= 1 \\
   x_{\text{MB},\text{P1}} + x_{\text{MB},\text{P2}} + x_{\text{MB},\text{P3}} &= 1 \\
   x_{\text{MC},\text{P1}} + x_{\text{MC},\text{P2}} + x_{\text{MC},\text{P3}} &= 1 \\
   \end{align*}

2. Each project is assigned to exactly one manager:
   \[
   \sum_{i \in I} x_{ij} = 1 \quad \forall j \in J
   \]
   Explicitly:
   \begin{align*}
   x_{\text{MA},\text{P1}} + x_{\text{MB},\text{P1}} + x_{\text{MC},\text{P1}} &= 1 \\
   x_{\text{MA},\text{P2}} + x_{\text{MB},\text{P2}} + x_{\text{MC},\text{P2}} &= 1 \\
   x_{\text{MA},\text{P3}} + x_{\text{MB},\text{P3}} + x_{\text{MC},\text{P3}} &= 1 \\
   \end{align*}

3. Binary variables:
   \[
   x_{ij} \in \{0,1\} \quad \forall i \in I,\, j \in J
   \]

##### Complete Model

\[
\begin{align*}
\min\quad & 3000\,x_{\text{MA},\text{P1}} + 3200\,x_{\text{MA},\text{P2}} + 3100\,x_{\text{MA},\text{P3}} \\
         & + 2800\,x_{\text{MB},\text{P1}} + 3300\,x_{\text{MB},\text{P2}} + 2900\,x_{\text{MB},\text{P3}} \\
         & + 2900\,x_{\text{MC},\text{P1}} + 3100\,x_{\text{MC},\text{P2}} + 3000\,x_{\text{MC},\text{P3}} \\
\text{s.t.}\quad
& x_{\text{MA},\text{P1}} + x_{\text{MA},\text{P2}} + x_{\text{MA},\text{P3}} = 1 \\
& x_{\text{MB},\text{P1}} + x_{\text{MB},\text{P2}} + x_{\text{MB},\text{P3}} = 1 \\
& x_{\text{MC},\text{P1}} + x_{\text{MC},\text{P2}} + x_{\text{MC},\text{P3}} = 1 \\
& x_{\text{MA},\text{P1}} + x_{\text{MB},\text{P1}} + x_{\text{MC},\text{P1}} = 1 \\
& x_{\text{MA},\text{P2}} + x_{\text{MB},\text{P2}} + x_{\text{MC},\text{P2}} = 1 \\
& x_{\text{MA},\text{P3}} + x_{\text{MB},\text{P3}} + x_{\text{MC},\text{P3}} = 1 \\
& x_{ij} \in \{0,1\} \quad \forall i \in \{\text{MA},\text{MB},\text{MC}\},\, j \in \{\text{P1},\text{P2},\text{P3}\}
\end{align*}
\]