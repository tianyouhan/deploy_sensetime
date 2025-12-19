from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar, PointPillar_Seg
from .pv_rcnn import PVRCNN
from .second_net import SECONDNet
from .second_net_iou import SECONDNetIoU
from .caddn import CaDDN
from .voxel_rcnn import VoxelRCNN
from .centerpoint import CenterPoint, CenterPoint_Seg
from .pv_rcnn_plusplus import PVRCNNPlusPlus
from .mppnet import MPPNet
from .mppnet_e2e import MPPNetE2E
from .pillarnet import PillarNet
from .voxelnext import VoxelNeXt
from .transfusion import TransFusion
from .bevfusion import BevFusion, BevFusion_Seg
from .bevfusion_qat import BevFusion_Seg_Qat
from .bevfusion_cp import BevFusion_cp
from .bevfusion_pointpillar import BevFusion_pp, BevFusion_pp_1
from .bevfusion_cp_1 import BevFusion_cp_1
from .bevfusion_cp_later_fs_multi_head import BevFusion_cp_later_fs_multi_head
from .bevfusion_depth_cp import BevFusion_depth_cp
from .pv_rcnn_plusplus_fusion import PVRCNNPlusPlus_fusion
from .bevfusion_pvbgop import BevFusion_PVBGOP
from .bevfusion_pvbgop_qat import BevFusion_PVBGOP_Qat
__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'SECONDNetIoU': SECONDNetIoU,
    'CaDDN': CaDDN,
    'VoxelRCNN': VoxelRCNN,
    'CenterPoint': CenterPoint,
    'PillarNet': PillarNet,
    'PVRCNNPlusPlus': PVRCNNPlusPlus,
    'MPPNet': MPPNet,
    'MPPNetE2E': MPPNetE2E,
    'PillarNet': PillarNet,
    'VoxelNeXt': VoxelNeXt,
    'TransFusion': TransFusion,
    'BevFusion': BevFusion,
    'BevFusion_cp': BevFusion_cp,
    'BevFusion_pp': BevFusion_pp,
    'BevFusion_pp_1': BevFusion_pp_1,
    'BevFusion_cp_1': BevFusion_cp_1,
    'BevFusion_cp_later_fs_multi_head': BevFusion_cp_later_fs_multi_head,
    'BevFusion_depth_cp': BevFusion_depth_cp,
    'PVRCNNPlusPlus_fusion': PVRCNNPlusPlus_fusion,

    'PointPillar_Seg': PointPillar_Seg,
    'CenterPoint_Seg': CenterPoint_Seg,
    'BevFusion_Seg': BevFusion_Seg,
    'BevFusion_PVBGOP': BevFusion_PVBGOP,
    'BevFusion_Seg_Qat': BevFusion_Seg_Qat,
    'BevFusion_PVBGOP_Qat': BevFusion_PVBGOP_Qat,
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
