from detectron2.config import CfgNode as CN

def add_decola_deta_config(cfg):
    _C = cfg
    _C.MODEL.DECOLA.DETA = CN()
    _C.MODEL.DECOLA.DETA.USE_DETA = False  
    _C.MODEL.DECOLA.DETA.ASSIGN_FIRST_STAGE = True 
    _C.MODEL.DECOLA.DETA.ASSIGN_SECOND_STAGE = True 