To formulate the mathematical model, I require the full cost matrix from cost_12x12.csv. However, since the data was not provided, I cannot retrieve the necessary parameters. Please provide the contents of cost_12x12.csv, and I will generate the complete mathematical model with all required vectors and matrices. 

If you provide the data, the model will be structured as follows:

##### Objective Function:

$\quad \min \sum_{i=1}^{12} \sum_{j=1}^{12} c_{ij} x_{ij}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{j=1}^{12} x_{ij} = 1 \quad \forall i \in \{1,2,\ldots,12\}$

$\sum_{i=1}^{12} x_{ij} = 1 \quad \forall j \in \{1,2,\ldots,12\}$

###### 2. Variable Constraints:

$x_{ij} \in \{0,1\}, \quad \forall i,j$

###### Retrieved Information

Please provide the cost matrix from cost_12x12.csv to complete the "Retrieved Information" section with all identifiers and values.