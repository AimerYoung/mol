import torch
from torch_geometric.data import InMemoryDataset

class PubChemDataset(InMemoryDataset):
    def __init__(self, path, text_max_len, prompt=None):
        super(PubChemDataset, self).__init__()
        self.data, self.slices = torch.load(path)

        self.path = path
        self.text_max_len = text_max_len
        
        if not prompt:
            self.prompt = 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. '
        else:
            self.prompt = prompt
        self.perm = None

    def __getitem__(self, index):
        if self.perm is not None:
            index = self.perm[index]
        data = self.get(index)
        smiles = data.smiles
        assert len(smiles.split('\n')) == 1

        text = data.text.split('\n')[:100]
        text = ' '.join(text) + '\n'
        
        words = data.words
        text_edge_index = data.text_edge_index
        return data, text, words, text_edge_index
    
    def shuffle(self):
        self.perm = torch.randperm(len(self)).tolist()
        return self

if __name__ == '__main__':
    dataset = PubChemDataset('./data/PubChem324kSP/train.pt', 512)
    print(dataset[0][0].keys())