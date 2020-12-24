import model
import torch
import os

from model_service.pytorch_model_service import PTServingBaseService


class PTVisionService(PTServingBaseService):

    def __init__(self, model_name, model_path):
        super(PTVisionService, self).__init__(model_name, model_path)
        self.model = model.Classifier()
        if torch.cuda.is_available():
            self.model.to(model.device)
        self.model.load_state_dict(torch.load(model_path, map_location=model.device))
        self.model.eval()
        dir_path = os.path.dirname(os.path.realpath(self.model_path))
        pca_path = os.path.join(dir_path, 'classifier-pca.pkl')
        self.preprocessor = model.Preprocessor(pca_path)
        self.postprocessor = model.Postprocessor()

    def _preprocess(self, data):
        return self.preprocessor(data)

    def _postprocess(self, data):
        return self.postprocessor(data)
