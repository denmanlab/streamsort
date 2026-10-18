"""Online (streaming) decoding: whitening, detection, template classification."""
from .whitening import Whitening, estimate_whitening
from .decoder import OnlineDecoder, DecoderParams, SpikeBatch, decode_window

__all__ = ["Whitening", "estimate_whitening", "OnlineDecoder", "DecoderParams",
           "SpikeBatch", "decode_window"]
