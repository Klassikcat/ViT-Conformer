from sagemaker_inference.transformer import Transformer
from sagemaker_inference.default_handler_service import DefaultHandlerService
from .inference_handler import VitConformerHandler


class ConformerHandlerService(DefaultHandlerService):
    def __init__(self):
        super(ConformerHandlerService, self).__init__()

    def initialize(self, context):
        super(ConformerHandlerService, self).initialize(context)

    def handle(self, data, context):
        pass
