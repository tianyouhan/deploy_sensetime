from .anchor_head_multi import AnchorHeadMulti
from .anchor_head_single import AnchorHeadSingle
from .anchor_head_template import AnchorHeadTemplate
from .point_head_box import PointHeadBox
from .point_head_simple import PointHeadSimple
from .point_intra_part_head import PointIntraPartOffsetHead
from .center_head import CenterHead
from .center_head_multi import MultiCenterHead
from .voxelnext_head import VoxelNeXtHead
from .transfusion_head import TransFusionHead
from .segmentation_head import SegHead, SegHead_pcseg, SegHead_pcseg_multihead
from .pp_heavy_head import pp_heavy_head
from .pp_heavy_head_qat import pp_heavy_head_qat
from .pp_heavy_head_pvbgop import pp_heavy_head_pvbgop
from .pp_heavy_head_pvbgop_qat import pp_heavy_head_pvbgop_qat
__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadSingle': AnchorHeadSingle,
    'PointIntraPartOffsetHead': PointIntraPartOffsetHead,
    'PointHeadSimple': PointHeadSimple,
    'PointHeadBox': PointHeadBox,
    'AnchorHeadMulti': AnchorHeadMulti,
    'CenterHead': CenterHead,
    'MultiCenterHead': MultiCenterHead,
    'VoxelNeXtHead': VoxelNeXtHead,
    'TransFusionHead': TransFusionHead,
    'pp_heavy_head': pp_heavy_head,
    'pp_heavy_head_qat': pp_heavy_head_qat,
    'pp_heavy_head_pvbgop': pp_heavy_head_pvbgop,
    'pp_heavy_head_pvbgop_qat': pp_heavy_head_pvbgop_qat,
    'SegHead': SegHead,
    'SegHead_pcseg': SegHead_pcseg,
    'SegHead_pcseg_multihead': SegHead_pcseg_multihead,
}
