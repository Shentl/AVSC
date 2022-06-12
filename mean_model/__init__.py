from .audio import MeanConcatDenseAudio
from .video import MeanConcatDenseVideo
from .early import MeanConcatDenseEarly
from .mid import MeanConcatDenseMid
from .decision import MeanConcatDenseDecision
from .decision_midattn import MeanConcatDenseDecisionMidAttn
from .audio_vattn import MeanAudioVattn
from .video_aattn import MeanVideoAattn
from .divide_t import DevideTimeMean
from .video_divide_t import MeanVDivideT
from .audio_divide_t import MeanADivideT
from .av_divide_t_mid import MeanAVDivideTMid
from .com_dt_t import MeanDTcomT


MODELS = {
    "audio": MeanConcatDenseAudio,
    "video": MeanConcatDenseVideo,
    "early": MeanConcatDenseEarly,
    "mid": MeanConcatDenseMid,
    "decision": MeanConcatDenseDecision,
    "dec_midattn": MeanConcatDenseDecisionMidAttn,
    "audio_vattn": MeanAudioVattn,
    "video_aattn": MeanVideoAattn,
    "divide_t": DevideTimeMean,
    "v_divide_t": MeanVDivideT,
    "a_divide_t": MeanADivideT,
    "av_divide_t_mid": MeanAVDivideTMid,
    "com_dt_t": MeanDTcomT
}


def load_model(name="mid"):
    assert name in MODELS.keys(), f"Model name can only be one of {MODELS.keys()}."
    print(f"Using model: '{name}'")
    return MODELS[name]
