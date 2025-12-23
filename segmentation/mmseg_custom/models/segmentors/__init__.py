# Copyright (c) OpenMMLab. All rights reserved.
# Commented out unused segmentors to avoid deprecated dependencies
# from .encoder_decoder_mask2former import EncoderDecoderMask2Former
# from .encoder_decoder_mask2former_aug import EncoderDecoderMask2FormerAug
# from .encoder_decoder_for_rdd import EncoderDecoderForRdd
from .encoder_decoder_for_teeth_single import EncoderDecoderForTeethSingle

__all__ = ['EncoderDecoderForTeethSingle']
