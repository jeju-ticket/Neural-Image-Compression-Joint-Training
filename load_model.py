from compressai.zoo.pretrained import load_pretrained
from compressai.zoo.image import cfgs, model_urls
from custom_model import CustomMeanScaleHyperprior
from compressai.models import (
    Cheng2020Anchor,
    Cheng2020Attention,
    FactorizedPrior,
    JointAutoregressiveHierarchicalPriors,
    ScaleHyperprior,
)
from torch.hub import load_state_dict_from_url

model_architectures = {
    "bmshj2018-factorized": FactorizedPrior,
    "bmshj2018-hyperprior": ScaleHyperprior,
    "mbt2018-mean": CustomMeanScaleHyperprior,  # changed
    "mbt2018": JointAutoregressiveHierarchicalPriors,
    "cheng2020-anchor": Cheng2020Anchor,
    "cheng2020-attn": Cheng2020Attention,
}

def load_model(
        architecture, metric, quality, pretrained=False, progress=True, **kwargs
):
    if architecture not in model_architectures:
        raise ValueError(f'Invalid architecture name "{architecture}"')

    if quality not in cfgs[architecture]:
        raise ValueError(f'Invalid quality value "{quality}"')

    if pretrained:
        if (
                architecture not in model_urls
                or metric not in model_urls[architecture]
                or quality not in model_urls[architecture][metric]
        ):
            raise RuntimeError("Pre-trained model not yet available")

        url = model_urls[architecture][metric][quality]
        state_dict = load_state_dict_from_url(url, progress=progress)
        state_dict = load_pretrained(state_dict)
        model = model_architectures[architecture].from_state_dict(state_dict)
        return model
    model = model_architectures[architecture](*cfgs[architecture][quality], **kwargs)
    return model
