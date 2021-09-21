import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig

# load model
DISTILBERT_PATH = "prajjwal1/bert-tiny"

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        
        self.config = AutoConfig.from_pretrained(DISTILBERT_PATH)
        self.config.update({"output_hidden_states":True, 
                    "hidden_dropout_prob": 0.0,
                    "attention_probs_dropout_prob" : 0,
                        "hidden_act" : "gelu_new",
                    "layer_norm_eps": 1e-7})   
        
        self.deberta = AutoModel.from_pretrained(DISTILBERT_PATH, config=self.config)

        self.ln = nn.LayerNorm(128)
        self.out = nn.Linear(128, 1)
        
    def forward(self, ids, mask):

        emb = self.deberta(ids, attention_mask=mask)['last_hidden_state'][:,0,:]
        output = self.ln(emb)
        logits = self.out(output)

        return logits

path = "DistilBert0.bin"
model = Model()
model.load_state_dict(torch.load(path, map_location = torch.device('cpu')))
model.eval()

def text_to_tensor(text):
    """
    Take a text axs input, tokenize it (including padding) and returns
    ids and mask.

    Parameters:
    ---------------------------
    text (str): string with the text to encode.

    """
    TOKENIZER = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    MAX_LEN = 256
    text = str(text)
    text = " ".join(text.split())
    inputs = TOKENIZER(text, add_special_tokens = True, max_length = MAX_LEN, padding=True, truncation=True)
    
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    
    padding_len = MAX_LEN-len(ids)
    ids = ids+([0]*padding_len)
    mask = mask+([0]*padding_len)

    return torch.tensor(ids, dtype=torch.long).unsqueeze(0), torch.tensor(mask, dtype=torch.long).unsqueeze(0)


def get_predictions(ids, mask):

    """
    Takes as input ids and mask of the text you want to make predictions on.

    Parameters:
    -------------------------
    ids (tensor): input ids
    mask (tensor): attention_mask
    """

    prediction = model(ids, mask)
    return prediction.item()

