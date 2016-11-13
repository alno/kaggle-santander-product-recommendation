import numpy as np

train_dates = ['2015-01-28', '2015-02-28', '2015-03-28', '2015-04-28', '2015-05-28', '2015-06-28', '2015-07-28', '2015-08-28', '2015-09-28', '2015-10-28', '2015-11-28', '2015-12-28', '2016-01-28', '2016-02-28', '2016-03-28', '2016-04-28', '2016-05-28']

test_date = '2016-06-28'


target_columns = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']


raw_data_dtypes = {
    'ncodpers': np.int64,  # Customer code
    'fecha_dato': np.str,  # Current date
    'fecha_alta': np.str,  # Date of first contract
    'ult_fec_cli_1t': np.str,  # Last date as primary customer (if he isn't at the end of the month)
    'age': np.str,  # Age
    'sexo': np.str,  # Sex
    'antiguedad': np.str,  # Customer seniority (in months)
    'canal_entrada': np.str,  # Channel used by the customer to join
    'cod_prov': np.float64,  # Province code (customer's address)
    'conyuemp': np.str,  # Spouse index. 1 if the customer is spouse of an employee
    'ind_actividad_cliente': np.float32,  # Activity index (1, active customer; 0, inactive customer)
    'ind_empleado': np.str,  # Employee index: A active, B ex employed, F filial, N not employee, P pasive
    'ind_nuevo': np.float32,  # New customer index. 1 if the customer registered in the last 6 months.
    'indext': np.str,  # Foreigner index (S (Yes) or N (No) if the customer's birth country is different than the bank country)
    'indfall': np.str,  # Deceased index. N/S
    'indrel': np.float64,  # 1 (First/Primary), 99 (Primary customer during the month but not at the end of the month)
    'indrel_1mes': np.str,  # Customer type at the beginning of the month ,1 (First/Primary customer), 2 (co-owner ),P (Potential),3 (former primary), 4(former co-owner)
    'indresi': np.str,  # Residence index (S (Yes) or N (No) if the residence country is the same than the bank country)
    'nomprov': np.str,  # Province name
    'pais_residencia': np.str,  # Customer's Country residence
    'renta': np.str,  # Gross income of the household
    'segmento': np.str,  # segmentation: 01 - VIP, 02 - Individuals 03 - college graduated
    'tipodom': np.float32,  # Addres type. 1, primary address
    'tiprel_1mes': np.str,  # Customer relation type at the beginning of the month, A (active), I (inactive), P (former customer),R (Potential)
    'ind_ahor_fin_ult1': np.uint8,
    'ind_aval_fin_ult1': np.uint8,
    'ind_cco_fin_ult1': np.uint8,
    'ind_cder_fin_ult1': np.uint8,
    'ind_cno_fin_ult1': np.uint8,
    'ind_ctju_fin_ult1': np.uint8,
    'ind_ctma_fin_ult1': np.uint8,
    'ind_ctop_fin_ult1': np.uint8,
    'ind_ctpp_fin_ult1': np.uint8,
    'ind_deco_fin_ult1': np.uint8,
    'ind_dela_fin_ult1': np.uint8,
    'ind_deme_fin_ult1': np.uint8,
    'ind_ecue_fin_ult1': np.uint8,
    'ind_fond_fin_ult1': np.uint8,
    'ind_hip_fin_ult1': np.uint8,
    'ind_nom_pens_ult1': np.float32,
    'ind_nomina_ult1': np.float32,
    'ind_plan_fin_ult1': np.uint8,
    'ind_pres_fin_ult1': np.uint8,
    'ind_reca_fin_ult1': np.uint8,
    'ind_recibo_ult1': np.uint8,
    'ind_tjcr_fin_ult1': np.uint8,
    'ind_valo_fin_ult1': np.uint8,
    'ind_viv_fin_ult1': np.uint8,
}
