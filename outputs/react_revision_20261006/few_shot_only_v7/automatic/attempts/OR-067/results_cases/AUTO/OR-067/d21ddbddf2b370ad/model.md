##### Objective Function:

$\quad \min \sum_{i \in \{\text{MA}, \text{MB}, \text{MC}\}} \sum_{j \in \{\text{P1}, \text{P2}, \text{P3}\}} c_{ij} x_{ij}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{j \in \{\text{P1}, \text{P2}, \text{P3}\}} x_{ij} = 1 \quad \forall i \in \{\text{MA}, \text{MB}, \text{MC}\}$

$\sum_{i \in \{\text{MA}, \text{MB}, \text{MC}\}} x_{ij} = 1 \quad \forall j \in \{\text{P1}, \text{P2}, \text{P3}\}$

###### 2. Variable Constraints:

$x_{ij} \in \{0,1\} \quad \forall i \in \{\text{MA}, \text{MB}, \text{MC}\},\ j \in \{\text{P1}, \text{P2}, \text{P3}\}$

##### Data Mapping

- Managers: MA, MB, MC
- Projects: P1, P2, P3
- Cost coefficients $c_{ij}$ are given by the CSV columns:
    - $c_{\text{MA},\text{P1}} = 3000$, $c_{\text{MA},\text{P2}} = 3200$, $c_{\text{MA},\text{P3}} = 3100$
    - $c_{\text{MB},\text{P1}} = 2800$, $c_{\text{MB},\text{P2}} = 3300$, $c_{\text{MB},\text{P3}} = 2900$
    - $c_{\text{MC},\text{P1}} = 2900$, $c_{\text{MC},\text{P2}} = 3100$, $c_{\text{MC},\text{P3}} = 3000$