import numpy as np


def contingent_mapping(contingency_table, final_offset=None):
    ctable = contingency_table[['primary', 'contingent']] 

    if final_offset is None:
        final_offset = 1 + ctable['primary'].max()
    
    ctable = ctable.query('primary != 0 and contingent != 0')
    
    # Drop non-duplicated primary labels
    ctable = ctable[ctable['primary'].duplicated(keep=False)]
    
    ctable = ctable.sort_values(['primary', 'contingent']).reset_index(drop=True)
    ctable['final'] = ctable.index + final_offset
    ctable['final'] = ctable['final'].astype(ctable['primary'].dtype)
    
    def narrow_col(col):
        m = ctable[col].max()
        for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
            if m <= np.iinfo(dtype).max:
                return ctable[col].astype(dtype, copy=False)

        raise AssertionError()
    
    ctable['primary'] = narrow_col('primary')
    ctable['contingent'] = narrow_col('contingent')
    ctable['final'] = narrow_col('final')

    return ctable
