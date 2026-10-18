##### Index Set  

$$I = \{1,2,\dots ,n\}$$  

##### Parameters  

$$w_i : \text{Revenue of FAUX product } i$$  
$$d_i : \text{Demand of FAUX product } i$$  
$$c_i : \text{Initial inventory of FAUX product } i$$  

##### Decision Variables  

$$x_i : \text{Number of units of FAUX product } i \text{ to fulfill}, \quad i \in I$$  

##### Objective Function  

$$\quad \max \quad \sum_{i \in I} w_i \, x_i$$  

##### Constraints  

###### 1. Demand Constraints  

$$x_i \leq d_i, \quad \forall i \in I$$  

###### 2. Inventory Constraints  

$$x_i \leq c_i, \quad \forall i \in I$$  

###### 3. Variable Domain  

$$x_i \in \mathbb{Z}, \quad x_i \geq 0, \quad \forall i \in I$$  

##### Data Mapping  

- **Table ID**: `file_0_view_0`  
- **Columns**: `Product Name`, `Revenue`, `Demand`, `Initial Inventory`  
- **Filter**: `Product Name` contains the string `"FAUX"`  
- **Parameter Mapping**:  
  - `w_i` ← `Revenue`  
  - `d_i` ← `Demand`  
  - `c_i` ← `Initial Inventory`