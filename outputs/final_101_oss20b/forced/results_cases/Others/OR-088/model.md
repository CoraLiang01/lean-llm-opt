## Mathematical Model

**Sets**  
\(F=\{F_1,\dots ,F_{15}\}\) plants  
\(C=\{C_1,\dots ,C_{15}\}\) customers  

**Parameters**  
\(\text{fixed\_cost}_i\) opening cost of plant \(i\) (from *cost.csv*)  
\(\text{capacity}_i\) maximum units plant \(i\) can supply (from *cost.csv*)  
\(\text{tc}_{ij}\) per‑unit transport cost from plant \(i\) to customer \(j\) (from *cost.csv*)  
\(\text{demand}_j\) required units for customer \(j\) (from *demand.csv*)  

**Decision Variables**  
\(x_{ij}\ge 0\) units shipped from plant \(i\) to customer \(j\) (continuous)  
\(y_i\in\{0,1\}\) 1 if plant \(i\) is opened, 0 otherwise  

**Objective**  
\[
\min \;\sum_{i\in F}\text{fixed\_cost}_i\,y_i
      +\sum_{i\in F}\sum_{j\in C}\text{tc}_{ij}\,x_{ij}
\]

**Constraints**

1. **Demand satisfaction**  
\[
\sum_{i\in F}x_{ij}= \text{demand}_j \qquad \forall j\in C
\]

2. **Plant capacity**  
\[
\sum_{j\in C}x_{ij}\le \text{capacity}_i\,y_i \qquad \forall i\in F
\]

3. **Non‑negativity**  
\[
x_{ij}\ge 0 \qquad \forall i\in F,\; j\in C
\]

4. **Binary opening decisions**  
\[
y_i\in\{0,1\} \qquad \forall i\in F
\]

---

## Data Mapping

| Parameter | Source Table | Column(s) | Mapping Rule |
|-----------|--------------|-----------|--------------|
| \(\text{fixed\_cost}_i\) | `file_0_view_0` | `fixed_cost` | one value per plant row |
| \(\text{capacity}_i\) | `file_0_view_0` | `capacity` | one value per plant row |
| \(\text{tc}_{ij}\) | `file_0_view_0` | `C1`–`C15` | each column gives cost from plant \(i\) to customer \(j\) |
| \(\text{demand}_j\) | `file_1_view_0` | `demand` | one value per customer row |

*Index sets \(F\) and \(C\) are taken from the row identifiers in the respective tables.*