File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/customer_demand.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 18
Columns: ['customer', 'demand']
Parsed column types: {'customer': 'object', 'demand': 'int64'}
Preview only (first 10 rows):
customer demand
      C1   4415
      C2   5430
      C3     81
      C4    146
      C5  10638
      C6   1663
      C7    151
      C8    185
      C9   1917
     C10   4489
Full-file column statistics: {"customer": {"missing": 0, "unique_nonempty": 18}, "demand": {"missing": 0, "unique_nonempty": 18, "numeric_range": [70.0, 10638.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "customer", "matching_columns": [], "exact_matching_columns": 0}, {"term": "customer_demand.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "demand", "matching_columns": [], "exact_matching_columns": 0}, {"term": "supply_capacity.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/supply_capacity.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 18
Columns: ['Unnamed: 0', 'supply_capacity']
Parsed column types: {'Unnamed: 0': 'object', 'supply_capacity': 'int64'}
Preview only (first 10 rows):
Unnamed: 0 supply_capacity
        S1            5963
        S2             702
        S3             350
        S4           11483
        S5            6585
        S6           11330
        S7             207
        S8             788
        S9            6967
       S10              43
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 18}, "supply_capacity": {"missing": 0, "unique_nonempty": 18, "numeric_range": [43.0, 22260.0]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "customer_demand.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "supply_capacity", "matching_columns": [], "exact_matching_columns": 0}, {"term": "supply_capacity.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv", "matching_columns": [], "exact_matching_columns": 0}]

---

File: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/transportation_costs.csv
CSV delimiter: comma; first line consumed as header.
Total rows: 18
Columns: ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18']
Parsed column types: {'Unnamed: 0': 'object', 'C1': 'float64', 'C2': 'float64', 'C3': 'float64', 'C4': 'float64', 'C5': 'float64', 'C6': 'float64', 'C7': 'float64', 'C8': 'float64', 'C9': 'float64', 'C10': 'float64', 'C11': 'float64', 'C12': 'float64', 'C13': 'float64', 'C14': 'float64', 'C15': 'float64', 'C16': 'float64', 'C17': 'float64', 'C18': 'float64'}
Preview only (first 10 rows):
Unnamed: 0                   C1                 C2                   C3                 C4                  C5                     C6                 C7                  C8                 C9                C10                C11                C12                C13                  C14                   C15                C16                 C17                 C18
        S1   159.83765495858208   6.42633790337285    7.582243029047409  99.52254454692138  135.16829918665007      7.590909409397181 2.3654667052579508   7.575274779934459  991.7798969083235 11.464155019446487  69.35418474260145 135.88274162390056 46.861565603907316    7.591284332304382    159.41034277497945  173.2178262613133   7.560918384716748  7.6626195540288755
        S2  0.10655701710223901   8.54894494648944 0.056878930310209665 212.07495813173495   1.040260758093917    0.06520831826866968  194.7110872056446  0.0494806010515883  839.3507562340868  89.40158875493893 217.73477041745696  68.76890997932357 141.20103295662906  0.06565023970186733     1.175025378373934 132.53399287398733   1.265678945608808 0.14381223458648024
        S3  0.07121132969220138  180.3996497782258  0.26229720469556855  182.8636027839307   1.735658687947286 0.00034579672599664544  9.336781626424907 0.28720505975061933 39.907671487452774  4.201696016865375 187.79672431983445 3.3050624901211525 142.30344029228047 0.012789665755568858 0.0031645198192960366 113.56966737977794 0.07213775163831262   1.724132015627177
        S4   1.5455382897983385 180.07836268028748 0.024056935459247156 182.81622784995758 0.09648380671738908    0.32805679124911646 167.94349941079716  0.3447071113014559  838.3178615319802  75.88119317030038  218.9576990348591 3.2896875372751215 165.76290434538876 0.015057752883055458  0.015504978004405388 132.22868668242745  0.0807047191286277  1.5358539138142018
        S5    16.41408384079902 20.385700354184078   16.413172756532582 478.03900704896546    296.930752792446       295.204481832446  535.0402811142731  295.47280243634304 29.543193182073455  259.9306953789963  477.9714035932095 16.957302922645596 476.97921893358273   295.19027061286835    344.40231658034236  311.0615468174578    345.603099976026   16.31794206958455
        S6    158.9313402279755 27.846407834093128    7.505708371284356  237.3730089947848   157.3399415011591      7.502126992568333  8.887771023202468  134.89750175293696  830.1122041518627 206.47987032908978 202.44057197951554  88.80510777650917  98.47373975154062    7.501579667106725    157.54235576369737  77.93428784134284   7.535086968463016   7.514549419003527
        S7  0.10720291412027193  154.1398544913415   0.8852806020632731 183.15061946024696  0.1196790346494559    0.03719608749792148 196.20925427585058  0.8877813480601408  718.5329214110352  4.219766244382137  219.2325360920747  58.92384746650274 142.10648477210282   0.7605337317228869    0.6661979296228859  6.272573102127474  1.9178600626840892 0.07215475483459184
        S8    2.220199373580392 227.15154482810524   47.959261803283134 220.05727166867416   41.14162586783218     48.081296729949585 215.21567259079265  41.310649904255584  804.8838110396675   47.1095086969049  203.7998107755676  5.591984830221476  9.653631540812063    48.09583337811525     41.21498273379365  8.431638784802882   2.247296057577594  2.2930186182029604
        S9 0.059394016843197485  180.7637642245803   0.0245032552848036  213.5491059653967   1.908865604731188     0.3778127261007191  9.350361928257238  0.5760978375171601  718.0228698036685  87.85660384116036  188.0412460602118  69.74234177397213 142.61913832558642  0.38169750882576514  0.018031149829208337 132.70438216743082  1.5265709751505647 0.07465438573576134
       S10    76.00470976255896  254.7550116921857     66.2904204966841  223.5475387800311   66.19918714920598     3.6895780121494055 193.92061614610816   66.49006559646251  37.59695422866174 46.746359383618476 11.812459372090945 146.53400490084056 224.10127986124817    66.42589614744958     77.48419913522422  9.836931402779376   76.49877965864712  3.6985586217330395
Full-file column statistics: {"Unnamed: 0": {"missing": 0, "unique_nonempty": 18}, "C1": {"missing": 0, "unique_nonempty": 18, "numeric_range": [0.0593940168431974, 1349.7946298063491]}, "C2": {"missing": 0, "unique_nonempty": 18, "numeric_range": [3.096409145258592, 1031.0578018743602]}, "C3": {"missing": 0, "unique_nonempty": 18, "numeric_range": [0.0240569354592471, 1348.9789699758965]}, "C4": {"missing": 0, "unique_nonempty": 18, "numeric_range": [10.0142344242029, 1179.2355621422678]}, "C5": {"missing": 0, "unique_nonempty": 18, "numeric_range": [0.096483806717389, 1347.625644089478]}, "C6": {"missing": 0, "unique_nonempty": 18, "numeric_range": [0.0003457967259966, 764.1106991806364]}, "C7": {"missing": 0, "unique_nonempty": 18, "numeric_range": [2.3654667052579508, 870.0007174209971]}, "C8": {"missing": 0, "unique_nonempty": 18, "numeric_range": [0.0494806010515883, 1156.114313238807]}, "C9": {"missing": 0, "unique_nonempty": 18, "numeric_range": [0.0225819370919379, 1151.4600713515997]}, "C10": {"missing": 0, "unique_nonempty": 18, "numeric_range": [4.144028420494834, 1433.228132588181]}, "C11": {"missing": 0, "unique_nonempty": 18, "numeric_range": [4.694117269797094, 1145.6184900389776]}, "C12": {"missing": 0, "unique_nonempty": 18, "numeric_range": [3.289687537275121, 820.8639813333887]}, "C13": {"missing": 0, "unique_nonempty": 18, "numeric_range": [7.9681413610932275, 1183.9344574977524]}, "C14": {"missing": 0, "unique_nonempty": 18, "numeric_range": [0.0127896657555688, 1156.3865347596354]}, "C15": {"missing": 0, "unique_nonempty": 18, "numeric_range": [0.003164519819296, 1156.383640245391]}, "C16": {"missing": 0, "unique_nonempty": 18, "numeric_range": [6.272573102127474, 759.8096548716817]}, "C17": {"missing": 0, "unique_nonempty": 18, "numeric_range": [0.0721377516383126, 718.6046793265715]}, "C18": {"missing": 0, "unique_nonempty": 18, "numeric_range": [0.0721547548345918, 1157.51304425127]}}
Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.
Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.
Query-name evidence: [{"term": "customer_demand.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "supply_capacity.csv", "matching_columns": [], "exact_matching_columns": 0}, {"term": "transportation_costs.csv", "matching_columns": [], "exact_matching_columns": 0}]