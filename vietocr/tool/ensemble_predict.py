from VietOCR.vietocr.tool.translate import build_model, ensemble_translate, translate_beam_search, process_input, predict
from VietOCR.vietocr.tool.utils import download_weights
from tqdm import tqdm
import torch
from collections import defaultdict 

class Predictor():
    def __init__(self, configs):

        device = configs[0]['device']
        _, vocab = build_model(configs[0])
        models = []
        weights = []
        for config in configs:
            model, _ = build_model(config)
            models.append(model)
            if config['weights'].startswith('http'):
                weights.append(download_weights(config['weights']))
            else:
                weights.append(config['weights'])

        for i in range(len(models)):
            models[i].load_state_dict(torch.load(weights[i], map_location=torch.device(device)))

        self.configs = configs
        self.models = models
        self.vocab = vocab
        self.device = device

    def predict(self, img, return_prob=False):
        img = process_input(img, self.config['dataset']['image_height'], 
                self.config['dataset']['image_min_width'], self.config['dataset']['image_max_width'])        
        img = img.to(self.config['device'])

        if self.config['predictor']['beamsearch']:
            sent = translate_beam_search(img, self.model)
            s = sent
            prob = None
        else:
            s, prob = ensemble_translate(img, self.model)
            s = s[0].tolist()
            prob = prob[0]

        s = self.vocab.decode(s)
        
        if return_prob:
            return s, prob
        else:
            return s

    def predict_batch(self, imgs, thresh_hold, ratio = 0.5, paddle_output_path = None, return_prob=False):
        bucket = defaultdict(list) # = [image_width: img_list]
        bucket_idx = defaultdict(list)
        bucket_pred = {}
        
        sents, probs = [0]*len(imgs), [0]*len(imgs)

        for i, img in enumerate(imgs):
            img = process_input(img, self.configs[0]['dataset']['image_height'], 
                self.configs[0]['dataset']['image_min_width'], self.configs[0]['dataset']['image_max_width'])        
        
            bucket[img.shape[-1]].append(img) #dict (tensor)
            bucket_idx[img.shape[-1]].append(i) #dict (list)

        for k, batch in tqdm(bucket.items(), desc="Infering: "):
            batch = torch.cat(batch, 0).to(self.device) #len batch: 186
            s, prob = ensemble_translate(batch, self.models, bucket_idx[k], thresh_hold, ratio, paddle_output_path)
            prob = prob.tolist()
            s = s.tolist()
            s = self.vocab.batch_decode(s)

            bucket_pred[k] = (s, prob)


        for k in bucket_pred:
            idx = bucket_idx[k]
            sent, prob = bucket_pred[k]
            for i, j in enumerate(idx):
                sents[j] = sent[i]
                probs[j] = prob[i]
        if return_prob: 
            return sents, probs
        else: 
            return sents

            