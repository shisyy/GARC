from copy import deepcopy

from nerfstudio.configs.method_configs import method_configs
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.plugins.types import MethodSpecification

from splart.splart import SplartModelConfig
from splart.splart_dataparser import SplartDataParserConfig
from splart.splart_pipeline import SplartPipelineConfig

splart_config = deepcopy(method_configs["splatfacto"])
splart_config.method_name = "splart"
splart_config.pipeline.datamanager.dataparser = SplartDataParserConfig()
splart_config.pipeline = SplartPipelineConfig(datamanager=splart_config.pipeline.datamanager, model=SplartModelConfig())
splart_config.optimizers.update(
    {
        "axis": {"optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15), "scheduler": None},
        "pivot": {"optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15), "scheduler": None},
        "dist": {"optimizer": AdamOptimizerConfig(lr=3e-5, eps=1e-15), "scheduler": None},
        "angle": {"optimizer": AdamOptimizerConfig(lr=3e-4, eps=1e-15), "scheduler": None},
        "mobilities": {"optimizer": AdamOptimizerConfig(lr=1e-1, eps=1e-15), "scheduler": None},
    }
)
splart_method = MethodSpecification(splart_config, "SplArt")
