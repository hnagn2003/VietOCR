import torch
import numpy as np
import math
from PIL import Image
from torch.nn.functional import log_softmax, softmax

from VietOCR.vietocr.model.transformerocr import VietOCR
from VietOCR.vietocr.model.vocab import Vocab
from VietOCR.vietocr.model.beam import Beam

pad2viet = [2,5,13,7,15,9,11,17,25,19,21,27,23,29,37,31,33,35
,39,41,43,45,47,49,51,57,55,53,59,61,69,63,65,67,71,73
,75,77,79,87,81,83,85,89,91,93,95,97,99,101,103,109,105,107
,111,113,121,115,117,119,123,125,133,127,131,129,135,137,139,141,143,145
,147,149,155,151,153,157,159,161,167,163,165,169,171,175,177,187,185,179
,181,183,173,189,4,12,6,14,8,10,16,18,24,20,22,26,28,36
,30,32,34,38,40,42,44,46,48,50,56,52,54,58,60,68,62,64
,66,70,72,74,76,78,86,80,82,84,88,90,92,94,96,98,100,102
,108,104,106,110,112,120,114,116,118,122,124,132,126,128,130,134,136,138
,140,142,144,146,148,154,150,152,156,158,160,166,162,164,168,170,176,178
,184,180,182,186,174,172,188,190]

viet2pad=[0,0,0,0,94,1,96,3,98,5,99,6,95,2,97,4,100,7
,101,9,103,10,104,12,102,8,105,11,106,13,108,15,109,16,110,17
,107,14,111,18,112,19,113,20,114,21,115,22,116,23,117,24,119,27
,120,26,118,25,121,28,122,29,124,31,125,32,126,33,123,30,127,34
,128,35,129,36,130,37,131,38,133,40,134,41,135,42,132,39,136,43
,137,44,138,45,139,46,140,47,141,48,142,49,143,50,145,52,146,53
,144,51,147,54,148,55,150,57,151,58,152,59,149,56,153,60,154,61
,156,63,157,65,158,64,155,62,159,66,160,67,161,68,162,69,163,70
,164,71,165,72,166,73,168,75,169,76,167,74,170,77,171,78,172,79
,174,81,175,82,173,80,176,83,177,84,185,92,184,85,178,86,179,89
,181,90,182,91,180,88,183,87,186,93,187]

def batch_translate_beam_search(img, model, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):
  # img: NxCxHxW
    model.eval()
    device = img.device
    sents = []

    with torch.no_grad():
        src = model.cnn(img)
        memories = model.transformer.forward_encoder(src)
        for i in range(src.size(0)):
            # memory = memories[:,i,:].repeat(1, beam_size, 1) # TxNxE
            memory = model.transformer.get_memory(memories, i)
            sent = beamsearch(memory, model, device, beam_size, candidates, max_seq_length, sos_token, eos_token)
            sents.append(sent)

    sents = np.asarray(sents)

    return sents
   
def translate_beam_search(img, model, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):
    # img: 1xCxHxW
    model.eval()
    device = img.device

    with torch.no_grad():
        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src) #TxNxE
        sent = beamsearch(memory, model, device, beam_size, candidates, max_seq_length, sos_token, eos_token)

    return sent
        
def beamsearch(memory, model, device, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):    
    # memory: Tx1xE
    model.eval()

    beam = Beam(beam_size=beam_size, min_length=0, n_top=candidates, ranker=None, start_token_id=sos_token, end_token_id=eos_token)

    with torch.no_grad():
#        memory = memory.repeat(1, beam_size, 1) # TxNxE
        memory = model.transformer.expand_memory(memory, beam_size)

        for _ in range(max_seq_length):
            
            tgt_inp = beam.get_current_state().transpose(0,1).to(device)  # TxN
            decoder_outputs, memory = model.transformer.forward_decoder(tgt_inp, memory)

            log_prob = log_softmax(decoder_outputs[:,-1, :].squeeze(0), dim=-1)
            beam.advance(log_prob.cpu())
            
            if beam.done():
                break
                
        scores, ks = beam.sort_finished(minimum=1)

        hypothesises = []
        for i, (times, k) in enumerate(ks[:candidates]):
            hypothesis = beam.get_hypothesis(times, k)
            hypothesises.append(hypothesis)
    
    return [1] + [int(i) for i in hypothesises[0][:-1]]

def translate(img, model, max_seq_length=128, sos_token=1, eos_token=2):
    "data: BxCXHxW"
    model.eval()
    device = img.device

    with torch.no_grad():
        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src)

        translated_sentence = [[sos_token]*len(img)]
        char_probs = [[1]*len(img)]

        max_length = 0

        while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T==eos_token, axis=1)):

            tgt_inp = torch.LongTensor(translated_sentence).to(device)
            
            output, memory = model.transformer.forward_decoder(tgt_inp, memory)
            output = softmax(output, dim=-1)
            output = output.to('cpu')

            values, indices  = torch.topk(output, 5)
            
            indices = indices[:, -1, 0]
            indices = indices.tolist()
            
            values = values[:, -1, 0]
            values = values.tolist()
            char_probs.append(values)

            translated_sentence.append(indices)   
            max_length += 1

            del output

        translated_sentence = np.asarray(translated_sentence).T
        
        char_probs = np.asarray(char_probs).T
        char_probs = np.multiply(char_probs, translated_sentence>3)
        char_probs = np.sum(char_probs, axis=-1)/(char_probs>0).sum(-1)
    
    return translated_sentence, char_probs


def build_model(config):
    vocab = Vocab(config['vocab'])
    device = config['device']
    
    model = VietOCR(len(vocab),
            config['backbone'],
            config['cnn'], 
            config['transformer'],
            config['seq_modeling'])
    
    model = model.to(device)

    return model, vocab

def resize(w, h, expected_height, image_min_width, image_max_width):
    new_w = int(expected_height * float(w) / float(h))
    round_to = 10
    new_w = math.ceil(new_w/round_to)*round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)

    return new_w, expected_height

def process_image(image, image_height, image_min_width, image_max_width):
    img = image.convert('RGB')

    w, h = img.size
    new_w, image_height = resize(w, h, image_height, image_min_width, image_max_width)

    img = img.resize((new_w, image_height), Image.ANTIALIAS)

    img = np.asarray(img).transpose(2,0, 1)
    img = img/255
    return img

def process_input(image, image_height, image_min_width, image_max_width):
    img = process_image(image, image_height, image_min_width, image_max_width)
    img = img[np.newaxis, ...]
    img = torch.FloatTensor(img)
    return img

def predict(filename, config):
    img = Image.open(filename)
    img = process_input(img)

    img = img.to(config['device'])

    model, vocab = build_model(config)
    s = translate(img, model)[0].tolist()
    s = vocab.decode(s)
    
    return s

def get_paddle_prob(img_ids, thresh_hold, paddle_output_path): #return tensor (batchsize, 26, 188)
    paddle_probs_path = paddle_output_path
    paddle_probs_file = np.load(paddle_probs_path)
    paddle_probs_list = []
    sentence_mark = []
    end_word_inx = 0
    for i in img_ids:
        tmp = paddle_probs_file[str(i + thresh_hold)] #(26,188)
        preds = tmp.argmax(axis=1)
        j = 0
        while (preds[j]==0):
            j = j+1
        beg_word_inx = min(j, 25)
        while (preds[j]!=0):
            j = j+1
        end_word_inx = min(j - 1, len(preds) - 1)
        sentence_mark.append([beg_word_inx, end_word_inx])
        # print("_---------------", sentence_mark, preds)

        if(beg_word_inx>23): #debug
            print("_------->23--------",i, sentence_mark, preds)
        tmp = torch.from_numpy(tmp)
        paddle_probs_list.append((tmp))
    return torch.stack(paddle_probs_list, dim=0), sentence_mark

def paddle2vietprob(char_paddle_prob):
    char_viet_prob = torch.zeros(191)
    for i in range(191):
        # print((viet2pad[i]))
        char_viet_prob[i] = char_paddle_prob[int(viet2pad[i])]
    return char_viet_prob

def ensemble_translate(img, models, img_ids, thresh_hold = 0, ratio=0.5, paddle_output_path = None,max_seq_length=128, sos_token=1, eos_token=2): #img_ids: [0, 6, 10]
    "data: BxCXHxW"
    paddle_probs, sentence_mark = get_paddle_prob(img_ids, thresh_hold, paddle_output_path) # (batchsize , 26 ,188), list(0: start mark, 1: end mark)
    for i in range(len(models)):
        models[i].eval()
    device = img.device
    with torch.no_grad():
        list_output = []
        srcs = []
        memorys = []
        for model in models:
            src = model.cnn(img)
            srcs.append(src)
            memorys.append(model.transformer.forward_encoder(src))
        translated_sentence = [[sos_token]*len(img)]
        char_probs = [[1]*len(img)]
        max_length = 0
        while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T==eos_token, axis=1)):
            tgt_inp = torch.LongTensor(translated_sentence).to(device)
        
            list_output = []
            for i in range(len(models)):
                output, memorys[i] = models[i].transformer.forward_decoder(tgt_inp, memorys[i])
                output = output.to('cpu')
                list_output.append(output[:,-1,:].unsqueeze(dim=1))

            final_output = torch.mean(torch.stack(list_output), dim=0)
            final_output = softmax(final_output, dim=-1)

            max_values, indices = torch.max(final_output, dim=2)
            for i in range(len(img)):
                if(int(indices[i]) > 3) and sentence_mark[i][0] <= sentence_mark[i][1]:
                    char_pos_in_paddle = sentence_mark[i][0]
                    char_paddle_prob = paddle_probs[i][char_pos_in_paddle] #torch size 188
                    char_viet_prob = paddle2vietprob(char_paddle_prob) #torch size 191
                    char_viet_prob = torch.unsqueeze(char_viet_prob, dim=0)
                    sentence_mark[i][0] = sentence_mark[i][0] + 1
                    final_output[i] = ratio*final_output[i] + (1-ratio)*char_viet_prob 
                
            max_values, indices = torch.max(final_output, dim=2)
            indices = indices.unsqueeze(2)

            values, indices  = torch.topk(final_output, 5)

            indices = indices[:, -1, 0]
            indices = indices.tolist()
            
            values = values[:, -1, 0]
            values = values.tolist()
            char_probs.append(values)
            translated_sentence.append(indices)   
            max_length += 1

            del output
        translated_sentence = np.asarray(translated_sentence).T
        
        char_probs = np.asarray(char_probs).T
        char_probs = np.multiply(char_probs, translated_sentence>3)
        char_probs = np.sum(char_probs, axis=-1)/(char_probs>0).sum(-1)

    return translated_sentence, char_probs
