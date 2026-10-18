##### Index Sets  

\(I\) = set of shelves  
\(J\) = set of products  

##### Parameters  

\(C_i\) – capacity of shelf \(i\) (from **capacity.csv**)  
\(v_j\) – value of product \(j\) (from **products.csv**)  
\(w_j\) – weight of product \(j\) (from **products.csv**)  

##### Variables  

\(x_{ij}\) – number of units of product \(j\) placed on shelf \(i\)  

##### Objective Function  

\[
\max \sum_{i \in I}\sum_{j \in J} v_j \cdot x_{ij}
\]

##### Constraints  

###### 1. Weight Capacity Constraints  

\[
\sum_{j \in J} w_j \cdot x_{ij} \leq C_i , \quad \forall i \in I
\]

###### 2. Variable Constraints  

\[
x_{ij} \in \mathbb{Z}_+ , \quad \forall i \in I,\; j \in J
\]

##### Data Mapping  

```json
{
  "file_0_view_0": {
    "Capacity": "C_i"
  },
  "file_1_view_0": {
    "Value": "v_j",
    "Weight": "w_j"
  }
}
```