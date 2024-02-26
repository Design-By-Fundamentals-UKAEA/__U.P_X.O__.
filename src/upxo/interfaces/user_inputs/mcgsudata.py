def _load_user_input_data_(xl_fname='input_dashboard.xls'):
    """
    CALL:
        from mcgs import _load_user_input_data_
        uidata = _load_user_input_data_(xl_fname='input_dashboard.xls')
    """
    workbook = xlrd.open_workbook('input_dashboard.xls')
    _sheet_, uidata = workbook.sheet_by_index(0), {}
    for r in range(_sheet_.nrows):
        cellname = _sheet_.cell_value(r, 0)
        cellvalue = _sheet_.cell_value(r, 1)
        uidata[cellname] = cellvalue
    return uidata