from dataclasses import dataclass


@dataclass
class dict_templates():
    # -----------------------------------------
    '''
    gc: Grain Centroids
    gcpos: Grain Centroids for position segregated grains
    rp: Representative Points
    jp2: Double Junction Points
    jp3: Triple Junction Points
    jp4: Qadruple Point Junctions
    '''
    mulpnt_gs2d = {'gc': None,
                   'gcpos': {'in': None, 'boundary': None,
                             'corner': None,
                             'left': None, 'bottom': None,
                             'right': None, 'top': None,
                             'pure_left': None, 'pure_bottom': None,
                             'pure_right': None, 'pure_top': None, },
                   'rp': None, 'jp2': None, 'jp3': None, }
    # -----------------------------------------
