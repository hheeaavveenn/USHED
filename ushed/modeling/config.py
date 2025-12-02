from cvpods.configs.fcos_config import FCOSConfig
from ushed.data import DUOMultiBranch  # noqa

from ushed.config import build_ushed_config


class USHEDConfig(FCOSConfig):
    def __init__(self):
        super().__init__()
        self._register_configuration(build_ushed_config())


config = USHEDConfig()
