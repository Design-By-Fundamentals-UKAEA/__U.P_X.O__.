class solat():
    '''
    Summary:
        Statistical Operations
    '''
    def __init__(self):
        pass
    def sp04(self, latticeArray):
        # sp04: sp0 to sp4
        latticeArray_04 = (latticeArray.min(), latticeArray.mean(), latticeArray.max(), latticeArray.std())
        return latticeArray_04