##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from distribution center $i$ (supplier) to customer group $j$ (demand), for all $i \in I$, $j \in J$.

##### Parameters

- $I = \{\text{supplier1}, \text{supplier2}, \text{supplier3}, \text{supplier4}, \text{supplier5}, \text{supplier6}, \text{supplier7}, \text{supplier8}\}$
- $J = \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$

- Customer group demands:
  - $d_{\text{demand1}} = 9$
  - $d_{\text{demand2}} = 66$
  - $d_{\text{demand3}} = 56$
  - $d_{\text{demand4}} = 17$
  - $d_{\text{demand5}} = 43$
  - $d_{\text{demand6}} = 62$
  - $d_{\text{demand7}} = 10$
  - $d_{\text{demand8}} = 37$

- Distribution center supply capacities:
  - $s_{\text{supplier1}} = 60$
  - $s_{\text{supplier2}} = 22$
  - $s_{\text{supplier3}} = 16$
  - $s_{\text{supplier4}} = 14$
  - $s_{\text{supplier5}} = 19$
  - $s_{\text{supplier6}} = 70$
  - $s_{\text{supplier7}} = 60$
  - $s_{\text{supplier8}} = 39$

- Transportation cost per unit $c_{ij}$ (from supplier $i$ to demand $j$):

|            | demand1   | demand2   | demand3   | demand4   | demand5   | demand6   | demand7   | demand8   |
|------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| supplier1  | 0.0302    | 229.5072  | 198.6236  | 12.9951   | 211.2073  | 134.9443  | 9.8222    | 11.3941   |
| supplier2  | 232.3469  | 3.6259    | 0.2861    | 45.7313   | 2.8305    | 107.0589  | 299.9632  | 23.7994   |
| supplier3  | 11.0619   | 0.2042    | 0.2789    | 45.7219   | 59.5490   | 5.0975    | 300.0012  | 23.7113   |
| supplier4  | 235.1795  | 43.7947   | 40.7098   | 0.0777    | 4.2377    | 131.7092  | 296.5559  | 29.8109   |
| supplier5  | 211.8581  | 47.6018   | 50.0401   | 86.1455   | 0.0620    | 5.3346    | 270.0629  | 3.8539    |
| supplier6  | 6.4551    | 88.1632   | 5.0471    | 151.4612  | 5.2908    | 0.0460    | 9.9367    | 103.7546  |
| supplier7  | 174.2723  | 250.5822  | 253.9041  | 16.2355   | 12.6431   | 175.0673  | 2.9838    | 317.0655  |
| supplier8  | 207.8701  | 1.5172    | 24.0272   | 27.1340   | 73.2067   | 125.7291  | 15.4631   | 0.2016    |

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:** Each customer group must receive exactly its demand.
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supply capacity:** Each distribution center cannot ship more than its supply capacity.
   \[
   \sum_{j \in J} x_{ij} \leq s_i, \quad \forall i \in I
   \]

3. **Nonnegativity:**
   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]

---

**All parameters, vectors, and matrices are included as retrieved from the CSV files.**