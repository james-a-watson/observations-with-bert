import transformers
import torch
    
class BERTClass(torch.nn.module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased', return_dict=False)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 29)
    
def load_ckp(): 
    raise NotImplementedError()

def train_model():
    raise NotImplementedError()