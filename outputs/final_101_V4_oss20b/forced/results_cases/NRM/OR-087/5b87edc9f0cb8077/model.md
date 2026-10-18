##### Index Sets  

$P = \{1,2,\dots ,80\}$  (product types)  

$D = \{1,2,\dots ,22\}$  (days in the month)  

##### Parameters  

$\displaystyle \text{demand}_p \quad (p\in P)$ – maximum demand (100 kg units) for product $p$  

$\displaystyle \text{price}_p \quad (p\in P)$ – selling price ($/100$ kg) for product $p$  

$\displaystyle \text{cost}_p \quad (p\in P)$ – production cost ($/100$ kg) for product $p$  

$\displaystyle \text{quota}_p \quad (p\in P)$ – maximum production quota per day (100 kg units) for product $p$  

$\displaystyle \text{actcost}_p \quad (p\in P)$ – fixed activation cost ($) for product $p$  

$\displaystyle \text{minbatch}_p \quad (p\in P)$ – minimum batch size (100 kg units) for product $p$  

##### Variables  

$\displaystyle x_{p,d}\in \mathbb{Z}_{\ge 0}\quad (p\in P,\;d\in D)$ – quantity of product $p$ produced on day $d$ (100 kg units)  

$\displaystyle y_p\in\{0,1\}\quad (p\in P)$ – binary activation decision for product $p$  

##### Objective Function  

$\displaystyle \max \;\sum_{p\in P}\sum_{d\in D}\bigl(\text{price}_p-\text{cost}_p\bigr)\,x_{p,d}\;-\;\sum_{p\in P}\text{actcost}_p\,y_p$  

##### Constraints  

###### 1. Demand Constraints  

$\displaystyle \sum_{d\in D}x_{p,d}\;\le\;\text{demand}_p,\qquad \forall p\in P$  

###### 2. Production Quota Constraints  

$\displaystyle x_{p,d}\;\le\;\text{quota}_p\,y_p,\qquad \forall p\in P,\;\forall d\in D$  

###### 3. Minimum Batch Size Constraints  

$\displaystyle \sum_{d\in D}x_{p,d}\;\ge\;\text{minbatch}_p\,y_p,\qquad \forall p\in P$  

###### 4. Activation Variable Domain  

$\displaystyle y_p\in\{0,1\},\qquad \forall p\in P$  

###### 5. Production Quantity Domain  

$\displaystyle x_{p,d}\in\mathbb{Z}_{\ge 0},\qquad \forall p\in P,\;\forall d\in D$  

##### Data Mapping  

| Parameter | Source Table | Column Name |
|-----------|--------------|-------------|
| $\text{demand}_p$ | file_0_view_0 | "Maximum Demand (100 kg units)" |
| $\text{price}_p$ | file_0_view_0 | "Selling Price ($/100 kg)" |
| $\text{cost}_p$ | file_0_view_0 | "Production Cost ($/100 kg)" |
| $\text{quota}_p$ | file_0_view_0 | "Production Quota (max per day)" |
| $\text{actcost}_p$ | file_1_view_0 | "Activation Cost ($)" |
| $\text{minbatch}_p$ | file_2_view_0 | "Minimum Batch Size (100 kg units)" |