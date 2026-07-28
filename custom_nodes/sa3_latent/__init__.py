import torch
import comfy.model_management


class EmptyLatentAudioSA3:
    """Empty audio latent shaped for Stable Audio 3.

    Core's EmptyLatentAudio is hardcoded to Stable Audio 1:

        length = round((seconds * 44100 / 2048) / 2) * 2
        torch.zeros([batch, 64, length])          # 64 channels
        {"downscale_ratio_temporal": 2048}

    But comfy/latent_formats.py declares:

        StableAudio1  latent_channels=64   temporal_downscale_ratio=2048
        StableAudio3  latent_channels=256  temporal_downscale_ratio=4096

    Feeding SA3 a latent built to SA1's geometry gives audio that starts
    coherent and then jumps around in time — it decodes as if the timeline
    were scrambled, because the decoder's samples-per-latent-step disagrees
    with how the latent was laid out.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seconds": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 1000.0, "step": 0.1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "channels": ("INT", {"default": 256, "min": 1, "max": 1024}),
                "downscale": ("INT", {"default": 4096, "min": 1, "max": 65536}),
                "sample_rate": ("INT", {"default": 44100, "min": 8000, "max": 192000}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "latent/audio"

    def generate(self, seconds, batch_size, channels, downscale, sample_rate):
        length = round((seconds * sample_rate / downscale) / 2) * 2
        latent = torch.zeros([batch_size, channels, length],
                             device=comfy.model_management.intermediate_device())
        return ({"samples": latent, "type": "audio",
                 "downscale_ratio_temporal": downscale},)


NODE_CLASS_MAPPINGS = {"EmptyLatentAudioSA3": EmptyLatentAudioSA3}
NODE_DISPLAY_NAME_MAPPINGS = {"EmptyLatentAudioSA3": "Empty Latent Audio (SA3)"}
