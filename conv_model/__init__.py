from .audio_1d import ConvConcatDenseAudio1d
from .audio import ConvConcatDenseAudio
from .audio_sed import ConvAudioSed
from .mid import ConvConcatDenseMid
from .decision import ConvConcatDenseDecision

# audio: 把channnel降到10，剩下的feature直接global_averge_pool到1维
# audio_1d: 4次2*2pooling后flatten(channel, t, f 全faltten了)
# 尝试将t分块做mean，然后flatten，而不是直接就整体mean了


MODELS = {
    "c_audio_1d":  ConvConcatDenseAudio1d,
    "c_audio": ConvConcatDenseAudio,
    "c_mid": ConvConcatDenseMid,
    "c_decision": ConvConcatDenseDecision,
    "audio_sed": ConvAudioSed
}


def load_model(name="mid"):
    assert name in MODELS.keys(), f"Model name can only be one of {MODELS.keys()}."
    print(f"Using model: '{name}'")
    return MODELS[name]
