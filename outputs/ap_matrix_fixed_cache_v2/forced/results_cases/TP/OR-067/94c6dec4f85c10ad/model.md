##### Decision Variables

Let $x_{ij} \in \{0,1\}$ for all managers $i \in \{\text{MA}, \text{MB}, \text{MC}\}$ and projects $j \in \{\text{P1}, \text{P2}, \text{P3}\}$, where $x_{ij} = 1$ if manager $i$ is assigned to project $j$, and $0$ otherwise.

##### Parameters

Let $c_{ij}$ be the cost for manager $i$ to complete project $j$, as given below:

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
\min \sum_{i \in \{\text{MA}, \text{MB}, \text{MC}\}} \sum_{j \in \{\text{P1}, \text{P2}, \text{P3}\}} c_{ij} x_{ij}
\]

##### Constraints

1. Each manager is assigned to exactly one project:
   \[
   \sum_{j \in \{\text{P1}, \text{P2}, \text{P3}\}} x_{ij} = 1 \quad \forall i \in \{\text{MA}, \text{MB}, \text{MC}\}
   \]

2. Each project is assigned to exactly one manager:
   \[
   \sum_{i \in \{\text{MA}, \text{MB}, \text{MC}\}} x_{ij} = 1 \quad \forall j \in \{\text{P1}, \text{P2}, \text{P3}\}
   \]

3. Binary assignment variables:
   \[
   x_{ij} \in \{0,1\} \quad \forall i \in \{\text{MA}, \text{MB}, \text{MC}\},\ j \in \{\text{P1}, \text{P2}, \text{P3}\}
   \]

##### Data

Managers: $\{\text{MA}, \text{MB}, \text{MC}\}$

Projects: $\{\text{P1}, \text{P2}, \text{P3}\}$

Costs:
- $c_{\text{MA},\text{P1}} = 3000$
- $c_{\text{MA},\text{P2}} = 3200$
- $c_{\text{MA},\text{P3}} = 3100$
- $c_{\text{MB},\text{P1}} = 2800$
- $c_{\text{MB},\text{P2}} = 3300$
- $c_{\text{MB},\text{P3}} = 2900$
- $c_{\text{MC},\text{P1}} = 2900$
- $c_{\text{MC},\text{P2}} = 3100$
- $c_{\text{MC},\text{P3}} = 3000$