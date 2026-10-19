##### Objective Function: 

$\quad \quad \min \sum_{i=1}^{8} \sum_{j=1}^{8} c_{ij} \cdot x_{ij}$

##### Constraints

###### 1. Supply Capacity Constraints: 

$\quad \quad \sum_{j=1}^{8} x_{ij} \leq S_i, \quad \forall i \in \{1,\dots,8\}$

###### 2. Demand Fulfillment Constraints: 

$\quad \quad \sum_{i=1}^{8} x_{ij} \geq D_j, \quad \forall j \in \{1,\dots,8\}$

###### 3. Non-negativity and Integrality Constraints: 

$\quad \quad x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i,j$

###### Retrieved Information

{
  "suppliers": [
    {"SupplierID":"1","SupplierName":"supplier1","SupplyCapacity":"60"},
    {"SupplierID":"2","SupplierName":"supplier2","SupplyCapacity":"22"},
    {"SupplierID":"3","SupplierName":"supplier3","SupplyCapacity":"16"},
    {"SupplierID":"4","SupplierName":"supplier4","SupplyCapacity":"14"},
    {"SupplierID":"5","SupplierName":"supplier5","SupplyCapacity":"19"},
    {"SupplierID":"6","SupplierName":"supplier6","SupplyCapacity":"70"},
    {"SupplierID":"7","SupplierName":"supplier7","SupplyCapacity":"60"},
    {"SupplierID":"8","SupplierName":"supplier8","SupplyCapacity":"39"}
  ],
  "customers": [
    {"CustomerID":"1","CustomerName":"demand1","Demand":"9"},
    {"CustomerID":"2","CustomerName":"demand2","Demand":"66"},
    {"CustomerID":"3","CustomerName":"demand3","Demand":"56"},
    {"CustomerID":"4","CustomerName":"demand4","Demand":"17"},
    {"CustomerID":"5","CustomerName":"demand5","Demand":"43"},
    {"CustomerID":"6","CustomerName":"demand6","Demand":"62"},
    {"CustomerID":"7","CustomerName":"demand7","Demand":"10"},
    {"CustomerID":"8","CustomerName":"demand8","Demand":"37"}
  ],
  "costs": [
    {"SupplierID":"1","demand1":"0.03020736643461065","demand2":"229.50723504640203","demand3":"198.62356558205792","demand4":"12.995050640153751","demand5":"211.20732124396406","demand6":"134.9442985029274","demand7":"9.822206398831067","demand8":"11.394077543225675"},
    {"SupplierID":"2","demand1":"232.34691308087835","demand2":"3.6258726438627473","demand3":"0.28605434149404785","demand4":"45.73127693242935","demand5":"2.8304796563034573","demand6":"107.05891033185472","demand7":"299.96317913389305","demand8":"23.79935436307657"},
    {"SupplierID":"3","demand1":"11.061938334356302","demand2":"0.2041995326579051","demand3":"0.2789447278030927","demand4":"45.721912724349636","demand5":"59.54895565737313","demand6":"5.097536739581239","demand7":"300.00118415135785","demand8":"23.711282707746893"},
    {"SupplierID":"4","demand1":"235.1794835706472","demand2":"43.794668963036194","demand3":"40.709846782945924","demand4":"0.07774496620087613","demand5":"4.237728183419554","demand6":"131.70915517494691","demand7":"296.55587567706743","demand8":"29.810940017561297"},
    {"SupplierID":"5","demand1":"211.85808746383796","demand2":"47.60180876530328","demand3":"50.04007716193931","demand4":"86.14548807358399","demand5":"0.06197897916874956","demand6":"5.3345515296262205","demand7":"270.06290423798396","demand8":"3.853933133973331"},
    {"SupplierID":"6","demand1":"6.45506633554524","demand2":"88.16323623354015","demand3":"5.047091671641611","demand4":"151.46120287365497","demand5":"5.290760161059401","demand6":"0.04602205335871525","demand7":"9.93670660180487","demand8":"103.75460989446313"},
    {"SupplierID":"7","demand1":"174.27229047340035","demand2":"250.58223528739327","demand3":"253.90413041857263","demand4":"16.235467318386764","demand5":"12.643140514778086","demand6":"175.0672824108511","demand7":"2.983839625303656","demand8":"317.0655193866389"},
    {"SupplierID":"8","demand1":"207.87006253790491","demand2":"1.517168471518212","demand3":"24.027239288137153","demand4":"27.133999276450346","demand5":"73.20672468851855","demand6":"125.72910359893308","demand7":"15.463103251642147","demand8":"0.20164987511903337"}
  ]
}