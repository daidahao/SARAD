from datetime import datetime, timedelta

scenarios = [
    {
        'name': 'AP01',
        'controller': 'P1-PC',
        'variables': [
            'SP1',
        ],
        'points': [
            'P1_B2016',
        ],
        'descriptions': [
            'Decrease or increase SP value of P1-PC. Restore as a form of a trapezoidal profile while hiding SP changes in HMI.',
        ],    
    },
    {
        'name': 'AP02',
        'controller': 'P1-PC',
        'variables': [
            'SP1',
            'PV1'
        ],
        'points': [
            'P1_B2016',
            'P1_PIT01',
        ],
        'descriptions': [
            'Decrease or increase SP value of P1-PC. Restore as a form of a trapezoidal profile while hiding SP changes in HMI.',
            'Attempt to maintain previous sensor value.',
        ],
    },
    {
        'name': 'AP03',
        'controller': 'P1-PC',
        'variables': [
            'SP1',
            'PV1',
            'PV2',
        ],
        'points': [
            'P1_B2016',
            'P1_PIT01',
            'P1_FIT01',
        ],
        'descriptions': [
            'Decrease or increase SP value of P1-PC. Restore as a form of a trapezoidal profile while hiding SP changes in HMI.',
            'Attempt to maintain previous sensor value.',
            'Attempt to maintain previous sensor value',
        ],
    },
    {
        'name': 'AP04',
        'controller': 'P1-PC',
        'variables': [
            'CV1',
        ],
        'points': [
            'P1_PCV01D',
        ],
        'descriptions': [
            'Decrease or increase CV value of P1-PC. Restore to normal.',
        ],
    },
    {
        'name': 'AP05',
        'controller': 'P1-PC',
        'variables': [
            'CV1',
            'PV1',
        ],
        'points': [
            'P1_PCV01D',
            'P1_PIT01',
        ],
        'descriptions': [
            'Decrease or increase CV value of P1-PC. Restore to normal.',
            'Attempt to maintain previous sensor value.',
        ],
    },
    {
        'name': 'AP06',
        'controller': 'P1-PC',
        'variables': [
            'SP1-ST',
        ],
        'points': [
            'P1_B2016',
        ],
        'descriptions': [
            'Short-term (ST) attack that decrease or increase SP value of P1-PC for a few seconds √ and restores to normal. Repeat several times while hiding SP changes in HMI.',
        ],
    },
    {
        'name': 'AP07',
        'controller': 'P1-PC',
        'variables': [
            'CV1-ST',
        ],
        'points': [
            'P1_PCV01D',
        ],
        'descriptions': [
            'Short-term (ST) attack that decrease or increase CV value of P1-PC for a few seconds and restores to normal. Repeat several times while hiding SP changes in HMI.',
        ],
    },
    {
        'name': 'AP08',
        'controller': 'P1-FC',
        'variables': [
            'SP1',
        ],
        'points': [
            'P1_B3005',
        ],
        'descriptions': [
            'Decrease or increase SP value of P1-FC. Restore as a form of a trapezoidal profile while hiding SP changes in HMI',
        ],
    },
    {
        'name': 'AP09',
        'controller': 'P1-FC',
        'variables': [
            'SP1',
            'PV1',
        ],
        'points': [
            'P1_B3005',
            'P1_FT03',
        ],
        'descriptions': [
            'Decrease or increase SP value of P1-FC. Restore as a form of a trapezoidal profile while hiding SP changes in HMI',
            'Attempt to maintain previous sensor value.',
        ],
    },
    {
        'name': 'AP10',
        'controller': 'P1-FC',
        'variables': [
            'SP1',
            'PV1',
            'PV2',
        ],
        'points': [
            'P1_B3005',
            'P1_FT03',
            'P1_LIT01',
        ],
        'descriptions': [
            'Decrease or increase SP value of P1-FC. Restore as a form of a trapezoidal profile while hiding SP changes in HMI',
            'Attempt to maintain previous sensor value.',
            'Attempt to maintain previous sensor value.',
        ],
    },
    {
        'name': 'AP11',
        'controller': 'P1-FC',
        'variables': [
            'CV1',
        ],
        'points': [
            'P1_FCV03D',
        ],
        'descriptions': [
            'Decrease or increase CV value of P1-FC. Restore in form of trapezoidal profile.',
        ],
    },
    {
        'name': 'AP12',
        'controller': 'P1-FC',
        'variables': [
            'CV1',
            'PV1',
        ],
        'points': [
            'P1_FCV03D',
            'P1_FT03',
        ],
        'descriptions': [
            'Decrease or increase CV value of P1-FC. Restore to normal.',
            'Attempt to maintain previous sensor value.',
        ],
    },
    {
        'name': 'AP13',
        'controller': 'P1-FC',
        'variables': [
            'CV1-ST',
        ],
        'points': [
            'P1_FCV03D',
        ],
        'descriptions': [
            'Short-term (ST) attack that decrease or increase CV value of P1-FC for a few seconds and restores to normal. Repeat several times while hiding SP changes in HMI.',
        ],
    },
    {
        'name': 'AP14',
        'controller': 'P1-LC',
        'variables': [
            'SP1',
        ],
        'points': [
            'P1_B3004',
        ],
        'descriptions': [
            'Decrease or increase SP value of P1-LC. Restore as a form of a trapezoidal profile while hiding SP changes in HMI.',
        ],
    },
    {
        'name': 'AP15',
        'controller': 'P1-LC',
        'variables': [
            'SP1',
            'PV1',
        ],
        'points': [
            'P1_B3004',
            'P1_LIT01',
        ],
        'descriptions': [
            'Decrease or increase SP value of P1-LC. Restore as a form of a trapezoidal profile while hiding SP changes in HMI.',
            'Attempt to repeat previous sensor value.',
        ],
    },
    {
        'name': 'AP16',
        'controller': 'P1-LC',
        'variables': [
            'CV1',
        ],
        'points': [
            'P1_LCV01D',
        ],
        'descriptions': [
            'Decrease or increase CV value of P1-LC. Restore to normal.',
        ],
    },
    {
        'name': 'AP17',
        'controller': 'P1-LC',
        'variables': [
            'CV1',
            'PV1',
        ],
        'points': [
            'P1_LCV01D',
            'P1_LIT01',
        ],
        'descriptions': [
            'Decrease or increase CV value of P1-LC. Restore to normal.',
            'Attempt to repeat previous sensor value.',
        ],
    },
    {
        'name': 'AP18',
        'controller': 'P1-LC',
        'variables': [
            'CV1-ST',
        ],
        'points': [
            'P1_LCV01D',
        ],
        'descriptions': [
            'Short-term (ST) attack that decrease or increase CV value of P1-LC for a few seconds and restores to normal. Repeat several times while hiding SP changes in HMI.',
        ],
    },
    {
        'name': 'AP19',
        'controller': 'P1-TC',
        'variables': [
            'CV1',
        ],
        'points': [
            'P1_FCV01D',
        ],
        'descriptions': [
            'Decrease or increase CV value of P1-TC. Restore to normal.',
        ],
    },
    {
        'name': 'AP20',
        'controller': 'P1-TC',
        'variables': [
            'CV1',
            'PV1',
        ],
        'points': [
            'P1_FCV01D',
            'P1_TIT01',
        ],
        'descriptions': [
            'Decrease or increase CV value of P1-TC. Restore to normal.',
            'Attempt to repeat previous sensor value.',
        ],
    },
    {
        'name': 'AP21',
        'controller': 'P1-TC',
        'variables': [
            'CV1-ST',
        ],
        'points': [
            'P1_FCV01D',
        ],
        'descriptions': [
            'Short-term (ST) attack that decrease or increase CV value of P1-TC for a few seconds and restores to normal. Repeat several times while hiding SP changes in HMI.',
        ],
    },
    {
        'name': 'AP22',
        'controller': 'P1-TC',
        'variables': [
            'SP1-LT',
        ],
        'points': [
            'P1_B4002',
        ],
        'descriptions': [
            'Long-term (LT) attack that decrease or increase SP value of P1-TC continuously for more than 10 minutes and restores to normal.',
        ],
    },
    {
        'name': 'AP23',
        'controller': 'P1-CC',
        'variables': [
            'CV1',
        ],
        'points': [
            'P1_PP04',
        ],
        'descriptions': [
            'Decrease or increase CV value of P1-CC. Restore to normal.',
        ],
    },
    {
        'name': 'AP24',
        'controller': 'P1-CC',
        'variables': [
            'CV1-ST',
        ],
        'points': [
            'P1_PP04',
        ],
        'descriptions': [
            'Short-term (ST) attack that decrease or increase CV value of P1-CC for a few second and restores to normal. Repeat several times while hiding SP changes in HMI.',
        ],
    },
    {
        'name': 'AP25',
        'controller': 'P1-CC',
        'variables': [
            'SP1-LT',
        ],
        'points': [
            'P1_PP04_SP',
        ],
        'descriptions': [
            'Long-term (LT) attack that decrease or increase SP value of P1-CC continuously for more than 10 minutes and restores to normal.',
        ],
    },
    {
        'name': 'AP26',
        'controller': 'P2-SC',
        'variables': [
            'SP1',
        ],
        'points': [
            'P2_AutoSD',
        ],
        'descriptions': [
            'Decrease or increase SP value of P2-SC. Restore as a form of a trapezoidal profile while hiding SP changes in HMI.',
        ],
    },
    {
        'name': 'AP27',
        'controller': 'P2-SC',
        'variables': [
            'SP1',
            'PV1',
        ],
        'points': [
            'P2_AutoSD',
            'P2_SIT01',
        ],
        'descriptions': [
            'Decrease or increase SP value of P2-SC. Restore as a form of a trapezoidal profile while hiding SP changes in HMI.',
            'Attempt to maintain previous sensor value.',
        ],
    },
    {
        'name': 'AP28',
        'controller': 'P2-SC',
        'variables': [
            'SP2',
        ],
        'points': [
            'P2_ManualSD',
        ],
        'descriptions': [
            'Decrease or increase SP value of P2-SC. Restore as a form of a trapezoidal profile while hiding SP changes in HMI.',
        ],
    },
    {
        'name': 'AP29',
        'controller': 'P2-SC',
        'variables': [
            'CV1',
        ],
        'points': [
            'P2_SCO',
        ],
        'descriptions': [
            'Decrease or increase CV value of P2-SC. Restore to normal.',
        ],
    },
    {
        'name': 'AP30',
        'controller': 'P2-SC',
        'variables': [
            'CV1',
            'PV1',
        ],
        'points': [
            'P2_SCO',
            'P2_SIT01',
        ],
        'descriptions': [
            'Decrease or increase CV value of P2-SC. Restore to normal.',
            'Attempt to maintain previous sensor value.',
        ],
    },
    {
        'name': 'AP31',
        'controller': 'P2-SC',
        'variables': [
            'SP1-ST',
        ],
        'points': [
            'P2_AutoSD',
        ],
        'descriptions': [
            'Short-term (ST) attack that decrease or increase CV value of P2-SC for a few seconds and restores to normal. Repeat several times while hiding SP changes in HMI.',
        ],
    },
    {
        'name': 'AP32',
        'controller': 'P2-TC',
        'variables': [
            'SP1',
        ],
        'points': [
            'P2_VTR01',
        ],
        'descriptions': [
            'Decrease or increase SP value of P2-TC. Restore as a form of a trapezoidal profile while hiding SP changes in HMI.',
        ],
    },
    {
        'name': 'AP33',
        'controller': 'P2-TC',
        'variables': [
            'SP2',
        ],
        'points': [
            'P2_VTR02',
        ],
        'descriptions': [
            'Decrease or increase SP value of P2-SC. Restore as a form of a trapezoidal profile while hiding SP changes in HMI.',
        ],
    },
    {
        'name': 'AP34',
        'controller': 'P2-TC',
        'variables': [
            'SP3',
        ],
        'points': [
            'P2_RTR',
        ],
        'descriptions': [
            'Decrease or increase SP value of P2-SC. Restore as a form of a trapezoidal profile while hiding SP changes in HMI.',
        ],
    },
    {
        'name': 'AP35',
        'controller': 'P3-LC',
        'variables': [
            'CV1',
        ],
        'points': [
            'P3_LCP01D',
        ],
        'descriptions': [
            'Attempt to repeat previous sensor value.',
        ],
    },
    {
        'name': 'AP36',
        'controller': 'P3-LC',
        'variables': [
            'CV1',
            'PV1',
        ],
        'points': [
            'P3_LCP01D',
            'P3_LIT01',
        ],
        'descriptions': [
            'Decrease or increase CV value of P3-LC. Restore to normal.',
            'Attempt to maintain previous sensor value.',
        ],
    },
    {
        'name': 'AP37',
        'controller': 'P3-LC',
        'variables': [
            'CV2',
        ],
        'points': [
            'P3_LCV01D',
        ],
        'descriptions': [
            'Decrease or increase CV value of P3-LC. Restore to normal.',
        ],
    },
    {
        'name': 'AP38',
        'controller': 'P3-LC',
        'variables': [
            'CV2',
            'PV1',
        ],
        'points': [
            'P3_LCV01D',
            'P3_LIT01',
        ],
        'descriptions': [
            'Decrease or increase CV value of P3-LC. Restore to normal.',
            'Attempt to maintain previous sensor value.',
        ],
    },
    {
        'name': 'AP39',
        'controller': 'P3-LC',
        'variables': [
            'CV2-LT',
        ],
        'points': [
            'P3_LCV01D',
        ],
        'descriptions': [
            'Long-term (LT) attack that decrease or increase CV value of P3-LC continuously for more than 10 minutes and restores to normal.',
        ],
    },
    {
        'name': 'AP40',
        'controller': 'P1-PC',
        'variables': [
            'SP1-LT',
        ],
        'points': [
            'P1_B2016',
        ],
        'descriptions': [
            'Long-term (LT) attack that decrease or increase SP value of P1-PC continuously for more than 10 minutes and restores to normal.',
        ],
    },
    {
        'name': 'AP41',
        'controller': 'P1-FC',
        'variables': [
            'SP1-LT',
        ],
        'points': [
            'P1_B3005',
        ],
        'descriptions': [
            'Long-term (LT) attack that decrease or increase SP value of P1-PC continuously for more than 10 minutes and restores to normal.',
        ],
    },
    {
        'name': 'AP42',
        'controller': 'P1-LC',
        'variables': [
            'CV1',
            'PV1',
            'PV2',
        ],
        'points': [
            'P1_LCV01D',
            'P1_LIT01',
            'P1_FT03',
        ],
        'descriptions': [
            'Decrease or increase CV value of P1-LC. Restore to normal.',
            'Attempt to repeat previous sensor value.',
            'Attempt to maintain previous sensor value.',
        ],
    },
    {
        'name': 'AP43',
        'controller': 'P1-LC',
        'variables': [
            'CV1-LT',
        ],
        'points': [
            'P1_LCV01D',
        ],
        'descriptions': [
            'Long-term (LT) attack that decrease or increase CV value of P1-LC continuously for more than 10 minutes and restores to normal.',
        ],
    },
    {
        'name': 'AP44',
        'controller': 'P1-LC',
        'variables': [
            'CV1-LT',
            'PV1-LT',
        ],
        'points': [
            'P1_LCV01D',
            'P1_LIT01',
        ],
        'descriptions': [
            'Decrease or increase CV value of P1-LC. Restore to normal.',
            'Attempt to repeat previous sensor value.',
        ],
    },
    {
        'name': 'AP45',
        'controller': 'P1-TC',
        'variables': [
            'SP1',
        ],
        'points': [
            'P1_B4002',
        ],
        'descriptions': [
            'Decrease or increase SP value of P1-TC. Restore as a form of a trapezoidal profile while hiding SP changes in HMI.',
        ],
    },
    {
        'name': 'AP46',
        'controller': 'P1-CC',
        'variables': [
            'CV1',
            'PV1',
        ],
        'points': [
            'P1_PP04',
            'P1_TIT03',
        ],
        'descriptions': [
            'Decrease or increase CV value of P1-CC. Resotre to normal.',
            'Attempt to repeat previous sensor value.',
        ],
    },
    {
        'name': 'AP47',
        'controller': 'P2-TC',
        'variables': [
            'SP2-LT',
        ],
        'points': [
            'P2_VTR02',
        ],
        'descriptions': [
            'Long-term (LT) attack that decrease or increase SP value of P2-TC continuously for more than 10 minutes and restores to normal.',
        ],
    },
]

attacks = [
    {
        'no': 1,
        'id': 'A101',
        'scenario': [
            'AP01',
        ],
        'controllers': [
            'P1-PC-SP1',
        ],
        'points': [
            'P1_B2016',
        ],
        'start_time': '7/7/2020 15:35',
        'duration': 192,
    },
    {
        'no': 2,
        'id': 'A102',
        'scenario': [
            'AP06',
        ],
        'controllers': [
            'P1-FC-SP1',
        ],
        'points': [
            'P1_B3005',
        ],
        'start_time': '7/7/2020 17:28',
        'duration': 98,
    },
    {
        'no': 3,
        'id': 'A103',
        'scenario': [
            'AP13',
        ],
        'controllers': [
            'P1-LC-CO1',
        ],
        'points': [
            'P1_LCV01D',
        ],
        'start_time': '7/7/2020 18:59',
        'duration': 190,
    },
    {
        'no': 4,
        'id': 'A104',
        'scenario': [
            'AP18',
        ],
        'controllers': [
            'P2-SC-CO1',
        ],
        'points': [
            'P2_SCO',
        ],
        'start_time': '7/7/2020 20:21',
        'duration': 60,
    },
    {
        'no': 5,
        'id': 'A105',
        'scenario': [
            'AP16',
        ],
        'controllers': [
            'P2-SC-SP1',
        ],
        'points': [
            'P2_AutoSD',
        ],
        'start_time': '7/7/2020 21:03',
        'duration': 89,
    },
    {
        'no': 6,
        'id': 'A201',
        'scenario': [
            'AP22',
        ],
        'controllers': [
            'P2-TC-SP2',
        ],
        'points': [
            'P2_VTR02',
        ],
        'start_time': '9/7/2020 15:47',
        'duration': 83,
    },
    {
        'no': 7,
        'id': 'A202',
        'scenario': [
            'AP02',
        ],
        'controllers': [
            'P1-PC-SP1PV1',
        ],
        'points': [
            'P1_B2016',
            'P1_PIT01',
        ],
        'start_time': '9/7/2020 17:38',
        'duration': 422,
    },
    {
        'no': 8,
        'id': 'A203',
        'scenario': [
            'AP15',
        ],
        'controllers': [
            'P1-LC-CO1-ST7',
        ],
        'points': [
            'P1_LCV01D',
        ],
        'start_time': '9/7/2020 18:59',
        'duration': 17,
    },
    {
        'no': 9,
        'id': 'A204',
        'scenario': [
            'AP07',
        ],
        'controllers': [
            'P1-FC-SP1PV1',
        ],
        'points': [
            'P1_B3005',
            'P1_FT03',
        ],
        'start_time': '9/7/2020 20:10',
        'duration': 259,
    },
    {
        'no': 10,
        'id': 'A205',
        'scenario': [
            'AP05',
        ],
        'controllers': [
            'P1-PC-SP1-ST10',
        ],
        'points': [
            'P1_B2016',
        ],
        'start_time': '9/7/2020 21:15',
        'duration': 123,
    },
    {
        'no': 11,
        'id': 'A206',
        'scenario': [
            'AP09',
        ],
        'controllers': [
            'P1-FC-CO1PV1',
        ],
        'points': [
            'P1_FCV03D',
            'P1_FT03',
        ],
        'start_time': '9/7/2020 23:02',
        'duration': 256,
    },
    {
        'no': 12,
        'id': 'A207',
        'scenario': [
            'AP21',
        ],
        'controllers': [
            'P2-TC-SP1',
        ],
        'points': [
            'P2_VTR01',
        ],
        'start_time': '10/7/2020 01:08',
        'duration': 68,
    },
    {
        'no': 13,
        'id': 'A208',
        'scenario': [
            'AP12',
        ],
        'controllers': [
            'P1-LC-SP1PV1',
        ],
        'points': [
            'P1_B3004',
            'P1_LIT01',
        ],
        'start_time': '10/7/2020 01:33',
        'duration': 261,
    },
    {
        'no': 14,
        'id': 'A209',
        'scenario': [
            'AP11',
        ],
        'controllers': [
            'P1-LC-SP1',
        ],
        'points': [
            'P1_B3004',
        ],
        'start_time': '10/7/2020 03:03',
        'duration': 159,
    },
    {
        'no': 15,
        'id': 'A210',
        'scenario': [
            'AP04',
        ],
        'controllers': [
            'P1-PC-CO1PV1',
        ],
        'points': [
            'P1_PCV01D',
            'P1_PITO1',
        ],
        'start_time': '10/7/2020 05:29',
        'duration': 421,
    },
    {
        'no': 16,
        'id': 'A211',
        'scenario': [
            'AP20',
        ],
        'controllers': [
            'P2-SC-SP1-ST5',
        ],
        'points': [
            'P2_AutoSD',
        ],
        'start_time': '10/7/2020 07:51',
        'duration': 45,
    },
    {
        'no': 17,
        'id': 'A212',
        'scenario': [
            'AP17',
        ],
        'controllers': [
            'P2-SC-SP1PV1',
        ],
        'points': [
            'P2_AutoSD',
            'P2_SIT01',
        ],
        'start_time': '10/7/2020 09:13',
        'duration': 152,
    },
    {
        'no': 18,
        'id': 'A213',
        'scenario': [
            'AP14',
        ],
        'controllers': [
            'P1-LC-CO1PV1',
        ],
        'points': [
            'P1_LCV01D',
            'P1_LIT01',
        ],
        'start_time': '10/7/2020 10:49',
        'duration': 254,
    },
    {
        'no': 19,
        'id': 'A214',
        'scenario': [
            'AP03',
        ],
        'controllers': [
            'P1-PC-CO1',
        ],
        'points': [
            'P1_PCV01D',
        ],
        'start_time': '10/7/2020 12:51',
        'duration': 152,
    },
    {
        'no': 20,
        'id': 'A215',
        'scenario': [
            'AP19',
        ],
        'controllers': [
            'P2-SC-CO1PV1',
        ],
        'points': [
            'P2_SCO',
            'P2_SIT01',
        ],
        'start_time': '10/7/2020 15:11',
        'duration': 151,
    },
    {
        'no': 21,
        'id': 'A216',
        'scenario': [
            'AP10',
        ],
        'controllers': [
            'P1-FC-CO1-ST10',
        ],
        'points': [
            'P1_FCV03D',
        ],
        'start_time': '10/7/2020 15:40',
        'duration': 65,
    },
    {
        'no': 22,
        'id': 'A217',
        'scenario': [
            'AP23',
        ],
        'controllers': [
            'P2-TC-SP3',
        ],
        'points': [
            'P2_RTR',
        ],
        'start_time': '10/7/2020 16:22',
        'duration': 184,
    },
    {
        'no': 23,
        'id': 'A218',
        'scenario': [
            'AP08',
        ],
        'controllers': [
            'P1-FC-CO1',
        ],
        'points': [
            'P1_FCV03D',
        ],
        'start_time': '10/7/2020 18:21',
        'duration': 99,
    },
    {
        'no': 24,
        'id': 'A219',
        'scenario': [
            'AP24',
        ],
        'controllers': [
            'P3-LC-CO1',
        ],
        'points': [
            'P3_LCP01D',
        ],
        'start_time': '10/7/2020 21:25',
        'duration': 119,
    },
    {
        'no': 25,
        'id': 'A220',
        'scenario': [
            'AP25',
        ],
        'controllers': [
            'P3-LC-CO2',
        ],
        'points': [
            'P2_LCV01D',
        ],
        'start_time': '10/7/2020 22:56',
        'duration': 119,
    },
    {
        'no': 26,
        'id': 'A301',
        'scenario': [
            'AP15',
            'AP06',
        ],
        'controllers': [
            'P1-LC-CO1-ST',
            'P1-FC-SP1',
        ],
        'points': [
            'P1_LCV01D',
            'P1_B3005',
        ],
        'start_time': '13/7/2020 13:51',
        'duration': 132,
    },
    {
        'no': 27,
        'id': 'A302',
        'scenario': [
            'AP02',
            'AP06',
        ],
        'controllers': [
            'P1-PC-SP1PV1',
            'P1-FC-SP1',
        ],
        'points': [
            'P1_B2016',
            'P1_PIT01',
            'P1_B3005',
        ],
        'start_time': '13/7/2020 15:21',
        'duration': 421,
    },
    {
        'no': 28,
        'id': 'A303',
        'scenario': [
            'AP03',
            'AP13',
        ],
        'controllers': [
            'P1-PC-CO1',
            'P1-LC-CO1',
        ],
        'points': [
            'P1_PCV01D',
            'P1_LCV01D',
        ],
        'start_time': '13/7/2020 18:11',
        'duration': 189,
    },
    {
        'no': 29,
        'id': 'A304',
        'scenario': [
            'AP16',
            'AP21',
        ],
        'controllers': [
            'P2-SC-SP1',
            'P2-TC-SP1',
        ],
        'points': [
            'P2_AutoSD',
            'P2_VTR01',
        ],
        'start_time': '13/7/2020 20:53',
        'duration': 106,
    },
    {
        'no': 30,
        'id': 'A305',
        'scenario': [
            'AP18',
            'AP22',
        ],
        'controllers': [
            'P2-SC-CO1',
            'P2-TC-SP2',
        ],
        'points': [
            'P2_SCO',
            'P2_VTR02',
        ],
        'start_time': '13/7/2020 21:23',
        'duration': 84,
    },
    {
        'no': 31,
        'id': 'A306',
        'scenario': [
            'AP01',
            'AP16',
        ],
        'controllers': [
            'P1-PC-SP1',
            'P2-SC-SP1',
        ],
        'points': [
            'P1_B2016',
            'P2_AutoSD',
        ],
        'start_time': '13/7/2020 23:55',
        'duration': 238,
    },
    {
        'no': 32,
        'id': 'A307',
        'scenario': [
            'AP08',
            'AP21',
        ],
        'controllers': [
            'P1-FC-CO1',
            'P2-TC-SP1',
        ],
        'points': [
            'P1_FCV03D',
            'P2_VTR01',
        ],
        'start_time': '14/7/2020 01:51',
        'duration': 110,
    },
    {
        'no': 33,
        'id': 'A308',
        'scenario': [
            'AP14',
            'AP20',
        ],
        'controllers': [
            'P1-LC-CO1PV1',
            'P2-SC-SP1-ST',
        ],
        'points': [
            'P1_LCV01D',
            'P1_LIT01',
            'P2_AutoSD',
        ],
        'start_time': '14/7/2020 03:53',
        'duration': 255,
    },
    {
        'no': 34,
        'id': 'A401',
        'scenario': [
            'AP03',
            'AP12',
        ],
        'controllers': [
            'P1-PC-CO1',
            'P1-LC-SP1PV1',
        ],
        'points': [
            'P1_PCV01D',
            'P1_B3004',
            'P1_LIT01',
        ],
        'start_time': '28/7/2020 12:43',
        'duration': 254,
    },
    {
        'no': 35,
        'id': 'A402',
        'scenario': [
            'AP07',
            'AP25',
        ],
        'controllers': [
            'P1-FC-SP1PV1',
            'P3-LC-CO2',
        ],
        'points': [
            'P1_B3005',
            'P1_FT03',
            'P2_LCV01D',
        ],
        'start_time': '28/7/2020 13:45',
        'duration': 262,
    },
    {
        'no': 36,
        'id': 'A403',
        'scenario': [
            'AP12',
            'AP25',
        ],
        'controllers': [
            'P1-LC-SP1PV1',
            'P3-LC-CO2',
        ],
        'points': [
            'P1_B3004',
            'P1_LIT01',
            'P2_LCV01D',
        ],
        'start_time': '28/7/2020 15:57',
        'duration': 263,
    },
    {
        'no': 37,
        'id': 'A404',
        'scenario': [
            'AP19',
            'AP14',
        ],
        'controllers': [
            'P2-SC-CO1PV1',
            'P1-LC-CO1PV1',
        ],
        'points': [
            'P2_SCO',
            'P2_SIT01',
            'P1_LCV01D',
            'P1_LIT01',
        ],
        'start_time': '28/7/2020 17:45',
        'duration': 258,
    },
    {
        'no': 38,
        'id': 'A405',
        'scenario': [
            'AP20',
            'AP25',
        ],
        'controllers': [
            'P2-SC-SP1-ST',
            'P3-LC-CO2',
        ],
        'points': [
            'P2_AutoSD',
            'P2_LCV01D',
        ],
        'start_time': '28/7/2020 20:47',
        'duration': 120,
    },
    {
        'no': 39,
        'id': 'A501',
        'scenario': [
            'AP03',
            'AP22',
        ],
        'controllers': [
            'P1-PC-CO1',
            'P2-TC-SP2',
        ],
        'points': [
            'P1_PCV01D',
            'P2_VTR02',
        ],
        'start_time': '30/7/2020 11:16',
        'duration': 172,
    },
    {
        'no': 40,
        'id': 'A502',
        'scenario': [
            'AP09',
            'AP18',
        ],
        'controllers': [
            'P1-FC-CO1PV1',
            'P2-SC-CO1',
        ],
        'points': [
            'P1_FCV03D',
            'P1_FT03',
            'P2_SCO',
        ],
        'start_time': '30/7/2020 13:30',
        'duration': 258,
    },
    {
        'no': 41,
        'id': 'A503',
        'scenario': [
            'AP12',
            'AP18',
        ],
        'controllers': [
            'P1-LC-SP1PV1',
            'P2-SC-CO1',
        ],
        'points': [
            'P1_B3004',
            'P1_LIT01',
            'P2_SCO',
        ],
        'start_time': '30/7/2020 16:05',
        'duration': 256,
    },
    {
        'no': 42,
        'id': 'A504',
        'scenario': [
            'AP08',
            'AP25',
        ],
        'controllers': [
            'P1-FC-CO1',
            'P3-LC-CO2',
        ],
        'points': [
            'P1_FCV03D',
            'P2_LCV01D',
        ],
        'start_time': '30/7/2020 17:45',
        'duration': 120,
    },
    {
        'no': 43,
        'id': 'A505',
        'scenario': [
            'AP11',
            'AP20',
        ],
        'controllers': [
            'P1-LC-SP1',
            'P2-SC-SP1-ST',
        ],
        'points': [
            'P1_B3004',
            'P2_AutoSD',
        ],
        'start_time': '30/7/2020 18:38',
        'duration': 203,
    },
    {
        'no': 44,
        'id': 'A506',
        'scenario': [
            'AP19',
            'AP25',
        ],
        'controllers': [
            'P2-SC-CO1PV1',
            'P3-LC-CO2',
        ],
        'points': [
            'P2_SCO',
            'P2_SIT01',
            'P2_LCV01D',
        ],
        'start_time': '30/7/2020 20:42',
        'duration': 153,
    },
    {
        'no': 45,
        'id': 'A507',
        'scenario': [
            'AP20',
            'AP21',
        ],
        'controllers': [
            'P2-SC-SP1-ST',
            'P2-TC-SP1',
        ],
        'points': [
            'P2_AutoSD',
            'P2_VTR01',
        ],
        'start_time': '30/7/2020 23:13',
        'duration': 79,
    },
    {
        'no': 46,
        'id': 'A508',
        'scenario': [
            'AP10',
            'AP15',
        ],
        'controllers': [
            'P1-FC-CO1-ST',
            'P1-LC-CO1-ST',
        ],
        'points': [
            'P1_FCV03D',
            'P1_LCV01D',
        ],
        'start_time': '31/7/2020 01:15',
        'duration': 51,
    },
    {
        'no': 47,
        'id': 'A509',
        'scenario': [
            'AP01',
            'AP03',
        ],
        'controllers': [
            'P1-PC-SP1',
            'P1-PC-CO1',
        ],
        'points': [
            'P1_B2016',
            'P1_PCV01D',
        ],
        'start_time': '31/7/2020 02:01',
        'duration': 241,
    },
    {
        'no': 48,
        'id': 'A510',
        'scenario': [
            'AP11',
            'AP14',
        ],
        'controllers': [
            'P1-LC-SP1',
            'P1-LC-CO1PV1',
        ],
        'points': [
            'P1_B3004',
            'P1_LCV01D',
            'P1_LIT01',
        ],
        'start_time': '31/7/2020 09:54',
        'duration': 262,
    },
    {
        'no': 49,
        'id': 'A511',
        'scenario': [
            'AP23',
            'AP25',
        ],
        'controllers': [
            'P2-TC-SP3',
            'P3-LC-CO2',
        ],
        'points': [
            'P2_RTR',
            'P2_LCV01D',
        ],
        'start_time': '31/7/2020 10:40',
        'duration': 120,
    },
    {
        'no': 50,
        'id': 'A512',
        'scenario': [
            'AP06',
            'AP09',
        ],
        'controllers': [
            'P1-FC-SP1',
            'P1-FC-CO1PV1',
        ],
        'points': [
            'P1_B3005',
            'P1_FCV03D',
            'P1_FT03',
        ],
        'start_time': '31/7/2020 11:21',
        'duration': 262,
    },
]

def find_scenario(name):
    for scenario in scenarios:
        if name == scenario['name']:
            return scenario

assert len(scenarios) == len(set(s['name'] for s in scenarios)), \
    'scenario names must be unique.'
for scenario in scenarios:
    assert len(scenario['points']) == len(scenario['variables']), scenario['name']
    assert len(scenario['points']) == len(scenario['descriptions']), scenario['name']

assert len(attacks) == len(set(a['id'] for a in attacks)), \
    'attack ids must be unique.'
for attack in attacks:
    assert len(attack['points']) >= len(attack['controllers']), attack['id']
    attack['start_time_dt'] = datetime.strptime(attack['start_time'], '%d/%m/%Y %H:%M')
    attack['end_time_dt'] = attack['start_time_dt'] + timedelta(seconds=attack['duration'])


if __name__ == '__main__':
    for a in attacks:
        print(a['id'], a['scenario'], \
              a['start_time_dt'], a['end_time_dt'], a['points'])



