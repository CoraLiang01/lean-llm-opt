##### Decision Variables

$x_j \in \mathbb{Z}_{\ge 0}$ for $j=1,\dots,101$  ( number of units of model HiFi$j$ produced per day )

##### Objective Function

$\displaystyle \min Z \;=\;\sum_{i=1}^{3}\Bigl(C_i-\sum_{j=1}^{101}t_{ij}\,x_j\Bigr)$

##### Constraints

1. **Workstation capacity** $\displaystyle \sum_{j=1}^{101}t_{ij}\,x_j \;\le\; C_i,\qquad \forall i=1,2,3$

2. **Non‑negativity and integrality** $\displaystyle x_j \in \mathbb{Z}_{\ge 0},\qquad \forall j=1,\dots,101$

---

##### Retrieved Information

**Models**  
$M=\{\text{HiFi}1,\text{HiFi}2,\dots,\text{HiFi}101\}$  

**Workstations**  
$W=\{W1,W2,W3\}$  

**Maintenance percentages**  
$m_1=10,\; m_2=14,\; m_3=12$  

**Effective capacities**  
$C_1=1440\,(1-0.10)=1296$  
$C_2=1440\,(1-0.14)=1233.6$  
$C_3=1440\,(1-0.12)=1267.2$  

**Processing time matrix $t_{ij}$**  

*Row 1 (Workstation 1)*  
$[6,4,6,7,6,6,8,9,6,7,1,2,4,7,3,8,3,2,4,5,8,3,2,3,9,7,3,5,7,6,2,1,5,6,5,1,7,9,8,3,3,8,2,3,3,8,9,2,3,4,2,9,2,1,8,8,4,4,6,1,6,5,3,5,1,6,6,5,3,4,3,8,1,2,3,8,4,4,2,8,4,5,1,6,4,1,3,8,3,3,3,6,7,6,2,1,8,9,6,7,9]$

*Row 2 (Workstation 2)*  
$[5,5,5,1,7,8,7,5,6,8,9,9,2,6,9,4,1,2,9,3,8,5,9,5,8,7,1,1,9,7,1,9,6,4,7,4,8,6,5,3,6,7,6,2,1,1,3,8,4,3,6,9,8,7,2,2,5,4,3,8,8,6,6,3,1,6,2,6,1,3,7,1,1,2,8,7,8,8,7,5,2,5,6,2,3,2,3,8,4,9,6,1,4,8,8,6,8,5,5,8,3]$

*Row 3 (Workstation 3)*  
$[4,6,5,2,6,5,3,3,4,8,6,3,3,3,7,8,3,8,1,5,3,8,5,8,4,8,6,7,9,5,3,6,3,3,3,8,4,6,3,8,3,7,5,3,1,8,9,6,6,4,7,1,9,9,3,9,6,5,7,8,9,9,8,5,4,4,3,3,8,8,2,4,9,6,7,6,7,3,1,7,6,4,3,5,7,6,3,5,2,2,9,3,6,9,7,2,4,5,8,1,6]$

These data fully specify the linear programming model for minimizing total idle time across the three workstations.