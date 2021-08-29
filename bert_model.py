from transformers import BertTokenizer, BertModel,BertConfig
import os
import random
import torch
global points
points = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def download_model():
    """
    Download Bert model
    :return:
    """
    SEED = 1234

    random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True


    if device == 'cpu':
        print('cpu')
    else:
        n_gpu = torch.cuda.device_count()
        print(torch.cuda.get_device_name(0))

    print(os.getcwd())
    config = BertConfig.from_pretrained(os.getcwd() + '/config.json')

    class MyBert(torch.nn.Module):
        def __init__(self, config):
            super(MyBert, self).__init__()
            self.h1 = BertModel(config=config)
            self.classifier = torch.nn.Linear(768, 3)

        def forward(self, input_ids, attention_mask):
            output_1 = self.h1(input_ids=input_ids, attention_mask=attention_mask)
            hidden_state = output_1[0]
            pooler = hidden_state[:, 0]
            output = self.classifier(pooler)
            return output

    model = MyBert(config)
    model.load_state_dict(torch.load(os.getcwd() + '/best_chekpoint.pt'))

    model.to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(os.getcwd())

    return model, tokenizer