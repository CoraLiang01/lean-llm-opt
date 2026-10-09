CSVQA_DATA = {'ignored_file_indices': [3, 7],
 'query': 'Plan air-conditioner placement at FC_EAST_HVAC. Each item_ref represents a model assigned to the storage '
          'area in location_id. Choose the integer quantities that maximize net value. Each area has its own volume '
          'limit and cannot borrow capacity from another area. The supplied tables describe the current plan; use '
          'their rows directly. The item rows give unit_benefit_cents and item_fee_cents: earn the former per unit and '
          'pay the latter once for any positive quantity. The usage amounts and capacity_ledger amounts are already in '
          'matching base units for each resource. Choose zero for unauthorized options. For any other option, choose '
          'zero or an integer from minimum_lot to maximum_order. Sum the signed capacity_ledger entries separately by '
          'resource; the total usage of that resource must stay within this amount. Across all options, each category '
          'must meet its lower and upper quantity limits; category activation_fee_cents is deducted once if used. The '
          'incompatible table forbids joint selection, while requires means positive quantity of item_ref needs '
          'positive quantity of prerequisite_ref, without proportional quantities. Each bundle row contributes '
          'bonus_cents once if both options are selected; if either is unselected or unauthorized the bonus is zero. '
          'Report the maximum net benefit in USD cents. All fixed fees and bonuses are in the same unit.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['item_a', 'item_b', 'bonus_cents'],
             'file_index': 0,
             'file_name': 'export_01.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 8,
             'records': [{'source_row': 0,
                          'values': {'bonus_cents': '271', 'item_a': 'Rd1e83c40d290', 'item_b': 'R14062bafad03'}},
                         {'source_row': 1,
                          'values': {'bonus_cents': '420', 'item_a': 'Rcabc56f3d592', 'item_b': 'R45c372fcda76'}},
                         {'source_row': 2,
                          'values': {'bonus_cents': '214', 'item_a': 'R53ffd925fbfa', 'item_b': 'R77b7948b94e4'}},
                         {'source_row': 3,
                          'values': {'bonus_cents': '210', 'item_a': 'R12315a4dcd90', 'item_b': 'R5415495b66bf'}},
                         {'source_row': 4,
                          'values': {'bonus_cents': '132', 'item_a': 'Rd16887582167', 'item_b': 'R99c4a58ed0e9'}},
                         {'source_row': 5,
                          'values': {'bonus_cents': '210', 'item_a': 'R5415495b66bf', 'item_b': 'Rcabc56f3d592'}},
                         {'source_row': 6,
                          'values': {'bonus_cents': '166', 'item_a': 'R45c372fcda76', 'item_b': 'R57ed61d46b1b'}},
                         {'source_row': 7,
                          'values': {'bonus_cents': '361', 'item_a': 'R53ffd925fbfa', 'item_b': 'R12315a4dcd90'}}],
             'returned_rows': 8,
             'role': 'bundle bonuses',
             'table_id': 'file_0_view_0'},
            {'columns': ['resource', 'entry', 'amount', 'unit'],
             'file_index': 1,
             'file_name': 'export_02.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 6,
             'records': [{'source_row': 0,
                          'values': {'amount': '212000', 'entry': 'opening', 'resource': 'AREA_B', 'unit': 'ml'}},
                         {'source_row': 1,
                          'values': {'amount': '233000', 'entry': 'opening', 'resource': 'AREA_A', 'unit': 'ml'}},
                         {'source_row': 2,
                          'values': {'amount': '211000', 'entry': 'opening', 'resource': 'AREA_C', 'unit': 'ml'}},
                         {'source_row': 3,
                          'values': {'amount': '-8000', 'entry': 'reservation', 'resource': 'AREA_B', 'unit': 'ml'}},
                         {'source_row': 4,
                          'values': {'amount': '-11000', 'entry': 'reservation', 'resource': 'AREA_C', 'unit': 'ml'}},
                         {'source_row': 5,
                          'values': {'amount': '-12000', 'entry': 'reservation', 'resource': 'AREA_A', 'unit': 'ml'}}],
             'returned_rows': 6,
             'role': 'resource capacity',
             'table_id': 'file_1_view_0'},
            {'columns': ['category', 'minimum_quantity', 'maximum_quantity', 'activation_fee_cents'],
             'file_index': 2,
             'file_name': 'export_03.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 4,
             'records': [{'source_row': 0,
                          'values': {'activation_fee_cents': '136',
                                     'category': 'G2',
                                     'maximum_quantity': '18',
                                     'minimum_quantity': '6'}},
                         {'source_row': 1,
                          'values': {'activation_fee_cents': '306',
                                     'category': 'G0',
                                     'maximum_quantity': '17',
                                     'minimum_quantity': '5'}},
                         {'source_row': 2,
                          'values': {'activation_fee_cents': '257',
                                     'category': 'G1',
                                     'maximum_quantity': '18',
                                     'minimum_quantity': '6'}},
                         {'source_row': 3,
                          'values': {'activation_fee_cents': '358',
                                     'category': 'G3',
                                     'maximum_quantity': '17',
                                     'minimum_quantity': '5'}}],
             'returned_rows': 4,
             'role': 'category constraints',
             'table_id': 'file_2_view_0'},
            {'columns': ['item_a', 'item_b'],
             'file_index': 4,
             'file_name': 'export_05.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 11,
             'records': [{'source_row': 0, 'values': {'item_a': 'R77b7948b94e4', 'item_b': 'R12315a4dcd90'}},
                         {'source_row': 1, 'values': {'item_a': 'R77b7948b94e4', 'item_b': 'Re3ce1b0e9e02'}},
                         {'source_row': 2, 'values': {'item_a': 'Re3ce1b0e9e02', 'item_b': 'R57ed61d46b1b'}},
                         {'source_row': 3, 'values': {'item_a': 'R547a94bb1e2c', 'item_b': 'R2caf5f536f7b'}},
                         {'source_row': 4, 'values': {'item_a': 'R77b7948b94e4', 'item_b': 'R14062bafad03'}},
                         {'source_row': 5, 'values': {'item_a': 'R99c4a58ed0e9', 'item_b': 'R77b7948b94e4'}},
                         {'source_row': 6, 'values': {'item_a': 'Rcabc56f3d592', 'item_b': 'R953de5726b93'}},
                         {'source_row': 7, 'values': {'item_a': 'R12315a4dcd90', 'item_b': 'R57ed61d46b1b'}},
                         {'source_row': 8, 'values': {'item_a': 'Rd1e83c40d290', 'item_b': 'R547a94bb1e2c'}},
                         {'source_row': 9, 'values': {'item_a': 'R77b7948b94e4', 'item_b': 'R2caf5f536f7b'}},
                         {'source_row': 10, 'values': {'item_a': 'R12315a4dcd90', 'item_b': 'R77b7948b94e4'}}],
             'returned_rows': 11,
             'role': 'incompatible pairs',
             'table_id': 'file_4_view_0'},
            {'columns': ['item_ref',
                         'category',
                         'authorized',
                         'minimum_lot',
                         'maximum_order',
                         'configuration_id',
                         'location_id',
                         'unit_benefit_cents',
                         'item_fee_cents'],
             'file_index': 5,
             'file_name': 'export_06.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 14,
             'records': [{'source_row': 0,
                          'values': {'authorized': '1',
                                     'category': 'G2',
                                     'configuration_id': 'CFG_03',
                                     'item_fee_cents': '442',
                                     'item_ref': 'Rd16887582167',
                                     'location_id': 'AREA_B',
                                     'maximum_order': '7',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '1161'}},
                         {'source_row': 1,
                          'values': {'authorized': '1',
                                     'category': 'G0',
                                     'configuration_id': 'CFG_02',
                                     'item_fee_cents': '228',
                                     'item_ref': 'Rf5c1f762749b',
                                     'location_id': 'AREA_B',
                                     'maximum_order': '7',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '979'}},
                         {'source_row': 2,
                          'values': {'authorized': '1',
                                     'category': 'G0',
                                     'configuration_id': 'CFG_01',
                                     'item_fee_cents': '149',
                                     'item_ref': 'Rdc2feb49418c',
                                     'location_id': 'AREA_A',
                                     'maximum_order': '8',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '1548'}},
                         {'source_row': 3,
                          'values': {'authorized': '1',
                                     'category': 'G2',
                                     'configuration_id': 'CFG_02',
                                     'item_fee_cents': '422',
                                     'item_ref': 'R2caf5f536f7b',
                                     'location_id': 'AREA_A',
                                     'maximum_order': '12',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '1464'}},
                         {'source_row': 4,
                          'values': {'authorized': '1',
                                     'category': 'G0',
                                     'configuration_id': 'CFG_01',
                                     'item_fee_cents': '307',
                                     'item_ref': 'R99c4a58ed0e9',
                                     'location_id': 'AREA_C',
                                     'maximum_order': '8',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '1410'}},
                         {'source_row': 5,
                          'values': {'authorized': '1',
                                     'category': 'G2',
                                     'configuration_id': 'CFG_01',
                                     'item_fee_cents': '304',
                                     'item_ref': 'Rcfec8c35a828',
                                     'location_id': 'AREA_C',
                                     'maximum_order': '7',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '1287'}},
                         {'source_row': 6,
                          'values': {'authorized': '1',
                                     'category': 'G2',
                                     'configuration_id': 'CFG_02',
                                     'item_fee_cents': '104',
                                     'item_ref': 'Rcca417551e0d',
                                     'location_id': 'AREA_B',
                                     'maximum_order': '12',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '1530'}},
                         {'source_row': 7,
                          'values': {'authorized': '1',
                                     'category': 'G3',
                                     'configuration_id': 'CFG_01',
                                     'item_fee_cents': '252',
                                     'item_ref': 'R7a87faff4029',
                                     'location_id': 'AREA_B',
                                     'maximum_order': '11',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '944'}},
                         {'source_row': 8,
                          'values': {'authorized': '1',
                                     'category': 'G1',
                                     'configuration_id': 'CFG_03',
                                     'item_fee_cents': '190',
                                     'item_ref': 'R050d45e3aa87',
                                     'location_id': 'AREA_B',
                                     'maximum_order': '10',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '762'}},
                         {'source_row': 9,
                          'values': {'authorized': '1',
                                     'category': 'G1',
                                     'configuration_id': 'CFG_01',
                                     'item_fee_cents': '201',
                                     'item_ref': 'R45c372fcda76',
                                     'location_id': 'AREA_B',
                                     'maximum_order': '8',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '1034'}},
                         {'source_row': 10,
                          'values': {'authorized': '1',
                                     'category': 'G1',
                                     'configuration_id': 'CFG_01',
                                     'item_fee_cents': '120',
                                     'item_ref': 'Rd2a478d62ea8',
                                     'location_id': 'AREA_A',
                                     'maximum_order': '11',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '730'}},
                         {'source_row': 11,
                          'values': {'authorized': '1',
                                     'category': 'G3',
                                     'configuration_id': 'CFG_01',
                                     'item_fee_cents': '196',
                                     'item_ref': 'Re3ce1b0e9e02',
                                     'location_id': 'AREA_A',
                                     'maximum_order': '12',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '1512'}},
                         {'source_row': 12,
                          'values': {'authorized': '1',
                                     'category': 'G1',
                                     'configuration_id': 'CFG_01',
                                     'item_fee_cents': '464',
                                     'item_ref': 'R53ffd925fbfa',
                                     'location_id': 'AREA_C',
                                     'maximum_order': '11',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '377'}},
                         {'source_row': 13,
                          'values': {'authorized': '1',
                                     'category': 'G1',
                                     'configuration_id': 'CFG_02',
                                     'item_fee_cents': '221',
                                     'item_ref': 'Rb96d2724c33e',
                                     'location_id': 'AREA_B',
                                     'maximum_order': '9',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '259'}}],
             'returned_rows': 14,
             'role': 'item options',
             'table_id': 'file_5_view_0'},
            {'columns': ['item_ref',
                         'category',
                         'authorized',
                         'minimum_lot',
                         'maximum_order',
                         'configuration_id',
                         'location_id',
                         'unit_benefit_cents',
                         'item_fee_cents'],
             'file_index': 6,
             'file_name': 'export_07.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 14,
             'records': [{'source_row': 0,
                          'values': {'authorized': '0',
                                     'category': 'G2',
                                     'configuration_id': 'CFG_03',
                                     'item_fee_cents': '334',
                                     'item_ref': 'R14062bafad03',
                                     'location_id': 'AREA_C',
                                     'maximum_order': '12',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '9000'}},
                         {'source_row': 1,
                          'values': {'authorized': '1',
                                     'category': 'G2',
                                     'configuration_id': 'CFG_02',
                                     'item_fee_cents': '170',
                                     'item_ref': 'Rd1e83c40d290',
                                     'location_id': 'AREA_C',
                                     'maximum_order': '12',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '730'}},
                         {'source_row': 2,
                          'values': {'authorized': '1',
                                     'category': 'G2',
                                     'configuration_id': 'CFG_01',
                                     'item_fee_cents': '146',
                                     'item_ref': 'R953de5726b93',
                                     'location_id': 'AREA_A',
                                     'maximum_order': '8',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '755'}},
                         {'source_row': 3,
                          'values': {'authorized': '0',
                                     'category': 'G0',
                                     'configuration_id': 'CFG_03',
                                     'item_fee_cents': '219',
                                     'item_ref': 'R57ed61d46b1b',
                                     'location_id': 'AREA_C',
                                     'maximum_order': '13',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '9000'}},
                         {'source_row': 4,
                          'values': {'authorized': '1',
                                     'category': 'G0',
                                     'configuration_id': 'CFG_01',
                                     'item_fee_cents': '213',
                                     'item_ref': 'R9fa17b3624a3',
                                     'location_id': 'AREA_B',
                                     'maximum_order': '12',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '911'}},
                         {'source_row': 5,
                          'values': {'authorized': '1',
                                     'category': 'G0',
                                     'configuration_id': 'CFG_02',
                                     'item_fee_cents': '407',
                                     'item_ref': 'R77b7948b94e4',
                                     'location_id': 'AREA_A',
                                     'maximum_order': '9',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '568'}},
                         {'source_row': 6,
                          'values': {'authorized': '0',
                                     'category': 'G0',
                                     'configuration_id': 'CFG_03',
                                     'item_fee_cents': '183',
                                     'item_ref': 'R5415495b66bf',
                                     'location_id': 'AREA_A',
                                     'maximum_order': '10',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '9000'}},
                         {'source_row': 7,
                          'values': {'authorized': '0',
                                     'category': 'G3',
                                     'configuration_id': 'CFG_02',
                                     'item_fee_cents': '266',
                                     'item_ref': 'Rc1af75b326dc',
                                     'location_id': 'AREA_B',
                                     'maximum_order': '8',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '9000'}},
                         {'source_row': 8,
                          'values': {'authorized': '1',
                                     'category': 'G3',
                                     'configuration_id': 'CFG_02',
                                     'item_fee_cents': '163',
                                     'item_ref': 'Rf1ec5c99c962',
                                     'location_id': 'AREA_A',
                                     'maximum_order': '12',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '573'}},
                         {'source_row': 9,
                          'values': {'authorized': '1',
                                     'category': 'G1',
                                     'configuration_id': 'CFG_02',
                                     'item_fee_cents': '384',
                                     'item_ref': 'R12315a4dcd90',
                                     'location_id': 'AREA_C',
                                     'maximum_order': '13',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '844'}},
                         {'source_row': 10,
                          'values': {'authorized': '1',
                                     'category': 'G3',
                                     'configuration_id': 'CFG_03',
                                     'item_fee_cents': '493',
                                     'item_ref': 'Rcabc56f3d592',
                                     'location_id': 'AREA_C',
                                     'maximum_order': '12',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '809'}},
                         {'source_row': 11,
                          'values': {'authorized': '1',
                                     'category': 'G3',
                                     'configuration_id': 'CFG_03',
                                     'item_fee_cents': '477',
                                     'item_ref': 'R93243a2d30f2',
                                     'location_id': 'AREA_A',
                                     'maximum_order': '10',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '1023'}},
                         {'source_row': 12,
                          'values': {'authorized': '1',
                                     'category': 'G1',
                                     'configuration_id': 'CFG_03',
                                     'item_fee_cents': '156',
                                     'item_ref': 'R45aa14bd7960',
                                     'location_id': 'AREA_A',
                                     'maximum_order': '12',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '1128'}},
                         {'source_row': 13,
                          'values': {'authorized': '1',
                                     'category': 'G3',
                                     'configuration_id': 'CFG_02',
                                     'item_fee_cents': '497',
                                     'item_ref': 'R547a94bb1e2c',
                                     'location_id': 'AREA_C',
                                     'maximum_order': '9',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '861'}}],
             'returned_rows': 14,
             'role': 'item options',
             'table_id': 'file_6_view_0'},
            {'columns': ['item_ref', 'prerequisite_ref'],
             'file_index': 8,
             'file_name': 'export_09.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 16,
             'records': [{'source_row': 0,
                          'values': {'item_ref': 'Rf5c1f762749b', 'prerequisite_ref': 'R53ffd925fbfa'}},
                         {'source_row': 1,
                          'values': {'item_ref': 'R77b7948b94e4', 'prerequisite_ref': 'Re3ce1b0e9e02'}},
                         {'source_row': 2,
                          'values': {'item_ref': 'Rd16887582167', 'prerequisite_ref': 'R547a94bb1e2c'}},
                         {'source_row': 3,
                          'values': {'item_ref': 'R2caf5f536f7b', 'prerequisite_ref': 'R953de5726b93'}},
                         {'source_row': 4,
                          'values': {'item_ref': 'R14062bafad03', 'prerequisite_ref': 'Rdc2feb49418c'}},
                         {'source_row': 5,
                          'values': {'item_ref': 'Rd1e83c40d290', 'prerequisite_ref': 'R9fa17b3624a3'}},
                         {'source_row': 6,
                          'values': {'item_ref': 'R57ed61d46b1b', 'prerequisite_ref': 'Re3ce1b0e9e02'}},
                         {'source_row': 7,
                          'values': {'item_ref': 'R5415495b66bf', 'prerequisite_ref': 'R99c4a58ed0e9'}},
                         {'source_row': 8,
                          'values': {'item_ref': 'R12315a4dcd90', 'prerequisite_ref': 'Rdc2feb49418c'}},
                         {'source_row': 9,
                          'values': {'item_ref': 'R050d45e3aa87', 'prerequisite_ref': 'R53ffd925fbfa'}},
                         {'source_row': 10,
                          'values': {'item_ref': 'R45aa14bd7960', 'prerequisite_ref': 'R7a87faff4029'}},
                         {'source_row': 11,
                          'values': {'item_ref': 'Rc1af75b326dc', 'prerequisite_ref': 'R53ffd925fbfa'}},
                         {'source_row': 12,
                          'values': {'item_ref': 'Rcabc56f3d592', 'prerequisite_ref': 'R99c4a58ed0e9'}},
                         {'source_row': 13,
                          'values': {'item_ref': 'Rf1ec5c99c962', 'prerequisite_ref': 'Re3ce1b0e9e02'}},
                         {'source_row': 14,
                          'values': {'item_ref': 'Rb96d2724c33e', 'prerequisite_ref': 'R45c372fcda76'}},
                         {'source_row': 15,
                          'values': {'item_ref': 'R93243a2d30f2', 'prerequisite_ref': 'R45c372fcda76'}}],
             'returned_rows': 16,
             'role': 'requires pairs',
             'table_id': 'file_8_view_0'},
            {'columns': ['item_ref', 'resource', 'amount', 'unit'],
             'file_index': 9,
             'file_name': 'export_10.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 14,
             'records': [{'source_row': 0,
                          'values': {'amount': '8000',
                                     'item_ref': 'Rdc2feb49418c',
                                     'resource': 'AREA_A',
                                     'unit': 'ml'}},
                         {'source_row': 1,
                          'values': {'amount': '3000',
                                     'item_ref': 'Rf1ec5c99c962',
                                     'resource': 'AREA_A',
                                     'unit': 'ml'}},
                         {'source_row': 2,
                          'values': {'amount': '8000',
                                     'item_ref': 'R953de5726b93',
                                     'resource': 'AREA_A',
                                     'unit': 'ml'}},
                         {'source_row': 3,
                          'values': {'amount': '4000',
                                     'item_ref': 'R45aa14bd7960',
                                     'resource': 'AREA_A',
                                     'unit': 'ml'}},
                         {'source_row': 4,
                          'values': {'amount': '7000',
                                     'item_ref': 'R93243a2d30f2',
                                     'resource': 'AREA_A',
                                     'unit': 'ml'}},
                         {'source_row': 5,
                          'values': {'amount': '9000',
                                     'item_ref': 'R7a87faff4029',
                                     'resource': 'AREA_B',
                                     'unit': 'ml'}},
                         {'source_row': 6,
                          'values': {'amount': '4000',
                                     'item_ref': 'Rd16887582167',
                                     'resource': 'AREA_B',
                                     'unit': 'ml'}},
                         {'source_row': 7,
                          'values': {'amount': '5000',
                                     'item_ref': 'Rb96d2724c33e',
                                     'resource': 'AREA_B',
                                     'unit': 'ml'}},
                         {'source_row': 8,
                          'values': {'amount': '8000',
                                     'item_ref': 'R050d45e3aa87',
                                     'resource': 'AREA_B',
                                     'unit': 'ml'}},
                         {'source_row': 9,
                          'values': {'amount': '3000',
                                     'item_ref': 'R9fa17b3624a3',
                                     'resource': 'AREA_B',
                                     'unit': 'ml'}},
                         {'source_row': 10,
                          'values': {'amount': '7000',
                                     'item_ref': 'R53ffd925fbfa',
                                     'resource': 'AREA_C',
                                     'unit': 'ml'}},
                         {'source_row': 11,
                          'values': {'amount': '7000',
                                     'item_ref': 'Rcabc56f3d592',
                                     'resource': 'AREA_C',
                                     'unit': 'ml'}},
                         {'source_row': 12,
                          'values': {'amount': '6000',
                                     'item_ref': 'R12315a4dcd90',
                                     'resource': 'AREA_C',
                                     'unit': 'ml'}},
                         {'source_row': 13,
                          'values': {'amount': '5000',
                                     'item_ref': 'R57ed61d46b1b',
                                     'resource': 'AREA_C',
                                     'unit': 'ml'}}],
             'returned_rows': 14,
             'role': 'item usage',
             'table_id': 'file_9_view_0'},
            {'columns': ['item_ref', 'resource', 'amount', 'unit'],
             'file_index': 10,
             'file_name': 'export_11.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 14,
             'records': [{'source_row': 0,
                          'values': {'amount': '7000',
                                     'item_ref': 'R77b7948b94e4',
                                     'resource': 'AREA_A',
                                     'unit': 'ml'}},
                         {'source_row': 1,
                          'values': {'amount': '8000',
                                     'item_ref': 'Re3ce1b0e9e02',
                                     'resource': 'AREA_A',
                                     'unit': 'ml'}},
                         {'source_row': 2,
                          'values': {'amount': '8000',
                                     'item_ref': 'Rd2a478d62ea8',
                                     'resource': 'AREA_A',
                                     'unit': 'ml'}},
                         {'source_row': 3,
                          'values': {'amount': '3000',
                                     'item_ref': 'R5415495b66bf',
                                     'resource': 'AREA_A',
                                     'unit': 'ml'}},
                         {'source_row': 4,
                          'values': {'amount': '4000',
                                     'item_ref': 'R2caf5f536f7b',
                                     'resource': 'AREA_A',
                                     'unit': 'ml'}},
                         {'source_row': 5,
                          'values': {'amount': '8000',
                                     'item_ref': 'Rc1af75b326dc',
                                     'resource': 'AREA_B',
                                     'unit': 'ml'}},
                         {'source_row': 6,
                          'values': {'amount': '9000',
                                     'item_ref': 'R45c372fcda76',
                                     'resource': 'AREA_B',
                                     'unit': 'ml'}},
                         {'source_row': 7,
                          'values': {'amount': '3000',
                                     'item_ref': 'Rf5c1f762749b',
                                     'resource': 'AREA_B',
                                     'unit': 'ml'}},
                         {'source_row': 8,
                          'values': {'amount': '4000',
                                     'item_ref': 'Rcca417551e0d',
                                     'resource': 'AREA_B',
                                     'unit': 'ml'}},
                         {'source_row': 9,
                          'values': {'amount': '9000',
                                     'item_ref': 'Rcfec8c35a828',
                                     'resource': 'AREA_C',
                                     'unit': 'ml'}},
                         {'source_row': 10,
                          'values': {'amount': '5000',
                                     'item_ref': 'Rd1e83c40d290',
                                     'resource': 'AREA_C',
                                     'unit': 'ml'}},
                         {'source_row': 11,
                          'values': {'amount': '3000',
                                     'item_ref': 'R14062bafad03',
                                     'resource': 'AREA_C',
                                     'unit': 'ml'}},
                         {'source_row': 12,
                          'values': {'amount': '7000',
                                     'item_ref': 'R547a94bb1e2c',
                                     'resource': 'AREA_C',
                                     'unit': 'ml'}},
                         {'source_row': 13,
                          'values': {'amount': '6000',
                                     'item_ref': 'R99c4a58ed0e9',
                                     'resource': 'AREA_C',
                                     'unit': 'ml'}}],
             'returned_rows': 14,
             'role': 'item usage',
             'table_id': 'file_10_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
import pandas as pd
import numpy as np
import sys
import re

def solve_problem(CSVQA_FRAMES):
    item_frames = [CSVQA_FRAMES['file_5_view_0'], CSVQA_FRAMES['file_6_view_0']]
    item_rows = []
    for frame in item_frames:
        for (idx, row) in frame.iterrows():
            item_rows.append({'item_ref': row['item_ref'], 'category': row['category'], 'authorized': int(row['authorized']), 'minimum_lot': int(row['minimum_lot']), 'maximum_order': int(row['maximum_order']), 'location_id': row['location_id'], 'unit_benefit_cents': int(row['unit_benefit_cents']), 'item_fee_cents': int(row['item_fee_cents'])})
    items = []
    item_data = {}
    for row in item_rows:
        i = row['item_ref']
        items.append(i)
        item_data[i] = row
    I_auth = [i for i in items if item_data[i]['authorized'] == 1]
    I_unauth = [i for i in items if item_data[i]['authorized'] == 0]
    cat_frame = CSVQA_FRAMES['file_2_view_0']
    categories = []
    catmin = {}
    catmax = {}
    catfee = {}
    for (idx, row) in cat_frame.iterrows():
        c = row['category']
        categories.append(c)
        catmin[c] = int(row['minimum_quantity'])
        catmax[c] = int(row['maximum_quantity'])
        catfee[c] = int(row['activation_fee_cents'])
    cap_frame = CSVQA_FRAMES['file_1_view_0']
    resource_caps = {}
    for (idx, row) in cap_frame.iterrows():
        r = row['resource']
        amt = float(row['amount'])
        resource_caps.setdefault(r, 0.0)
        resource_caps[r] += amt
    resources = list(resource_caps.keys())
    usage = {i: {r: 0.0 for r in resources} for i in items}
    for usage_table in ['file_9_view_0', 'file_10_view_0']:
        frame = CSVQA_FRAMES[usage_table]
        for (idx, row) in frame.iterrows():
            i = row['item_ref']
            r = row['resource']
            amt = float(row['amount'])
            if i in items and r in resources:
                usage[i][r] = amt
    bundle_frame = CSVQA_FRAMES['file_0_view_0']
    B = []
    bonus = {}
    for (idx, row) in bundle_frame.iterrows():
        i = row['item_a']
        j = row['item_b']
        b = (i, j)
        B.append(b)
        bonus[b] = int(row['bonus_cents'])
    incmp_frame = CSVQA_FRAMES['file_4_view_0']
    Q = []
    for (idx, row) in incmp_frame.iterrows():
        i = row['item_a']
        j = row['item_b']
        Q.append((i, j))
    req_frame = CSVQA_FRAMES['file_8_view_0']
    P = []
    for (idx, row) in req_frame.iterrows():
        i = row['item_ref']
        j = row['prerequisite_ref']
        P.append((i, j))
    items_by_cat = {c: [] for c in categories}
    for i in items:
        c = item_data[i]['category']
        if c in categories:
            items_by_cat[c].append(i)
    items_by_res = {r: [] for r in resources}
    for i in items:
        loc = item_data[i]['location_id']
        if loc in resources:
            items_by_res[loc].append(i)
    m = gp.Model('AC_Placement')
    m.Params.MIPGap = 0.0001
    quantity_vars = {}
    for i in items:
        if item_data[i]['authorized'] == 1:
            lb = 0
            ub = item_data[i]['maximum_order']
            quantity_vars[i] = m.addVar(lb=lb, ub=ub, vtype=gp.GRB.INTEGER, name=f'x_{i}')
        else:
            quantity_vars[i] = m.addVar(lb=0, ub=0, vtype=gp.GRB.INTEGER, name=f'x_{i}')
    activation_vars = {}
    for i in items:
        activation_vars[i] = m.addVar(vtype=gp.GRB.BINARY, name=f'y_{i}')
    category_activation_vars = {}
    for c in categories:
        category_activation_vars[c] = m.addVar(vtype=gp.GRB.BINARY, name=f'z_{c}')
    bundle_vars = {}
    for b in B:
        bundle_vars[b] = m.addVar(vtype=gp.GRB.BINARY, name=f'w_{b[0]}_{b[1]}')
    m.update()
    obj = gp.LinExpr()
    obj += gp.quicksum((item_data[i]['unit_benefit_cents'] * quantity_vars[i] for i in items))
    obj -= gp.quicksum((item_data[i]['item_fee_cents'] * activation_vars[i] for i in items))
    obj -= gp.quicksum((catfee[c] * category_activation_vars[c] for c in categories))
    obj += gp.quicksum((bonus[b] * bundle_vars[b] for b in B))
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    for i in items:
        if item_data[i]['authorized'] == 0:
            m.addConstr(quantity_vars[i] == 0, name=f'unauth_{i}')
            m.addConstr(activation_vars[i] == 0, name=f'unauth_y_{i}')
        else:
            minlot = item_data[i]['minimum_lot']
            maxorder = item_data[i]['maximum_order']
            m.addConstr(quantity_vars[i] <= maxorder * activation_vars[i], name=f'link_ub_{i}')
            m.addConstr(quantity_vars[i] >= minlot * activation_vars[i], name=f'link_lb_{i}')
    for r in resources:
        m.addConstr(gp.quicksum((usage[i][r] * quantity_vars[i] for i in items_by_res[r])) <= resource_caps[r], name=f'cap_{r}')
    for c in categories:
        m.addConstr(gp.quicksum((quantity_vars[i] for i in items_by_cat[c])) >= catmin[c], name=f'catmin_{c}')
        m.addConstr(gp.quicksum((quantity_vars[i] for i in items_by_cat[c])) <= catmax[c], name=f'catmax_{c}')
        m.addConstr(gp.quicksum((quantity_vars[i] for i in items_by_cat[c])) <= catmax[c] * category_activation_vars[c], name=f'cat_z_ub_{c}')
        minlot_in_cat = min([item_data[i]['minimum_lot'] for i in items_by_cat[c]]) if items_by_cat[c] else 0
        m.addConstr(gp.quicksum((quantity_vars[i] for i in items_by_cat[c])) >= minlot_in_cat * category_activation_vars[c], name=f'cat_z_lb_{c}')
    for (i, j) in Q:
        if i in items and j in items:
            m.addConstr(activation_vars[i] + activation_vars[j] <= 1, name=f'incmp_{i}_{j}')
    for (i, j) in P:
        if i in items and j in items:
            m.addConstr(activation_vars[i] <= activation_vars[j], name=f'req_{i}_{j}')
    for b in B:
        (i, j) = b
        if i in items and j in items:
            m.addConstr(bundle_vars[b] <= activation_vars[i], name=f'bundle1_{i}_{j}')
            m.addConstr(bundle_vars[b] <= activation_vars[j], name=f'bundle2_{i}_{j}')
            m.addConstr(bundle_vars[b] >= activation_vars[i] + activation_vars[j] - 1, name=f'bundle3_{i}_{j}')
    m.update()
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)