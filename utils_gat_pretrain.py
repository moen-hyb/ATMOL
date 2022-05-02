import os

from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
import torch
import pandas as pd
from creat_data_DC import smile_to_graph

"""
预训练数据处理
"""


class TestbedDataset(InMemoryDataset):
    def __init__(self, root='tmp', dataset='', patt= 're',transform=None,
                 pre_transform=None, smile_graph=None):
        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.patt = patt
        self.processed_paths[0] = self.processed_paths[0] + self.patt

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(root, self.dataset)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + self.patt + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, root, dataset):
        data_list = []
        compound_iso_smiles = []
        df = pd.read_csv('data/' + root + '/data/' + dataset + '.csv')
        compound_iso_smiles += list(df['smiles'])
        compound_iso_smiles = set(compound_iso_smiles)

        count = 0
        for smile in compound_iso_smiles:
            if len(smile) < 4:
                continue
            count = count + 1
            # smile = 'CCCCSc1nc(N)c2ncn(Cc3c(F)ccc(C)c3F)c2n1'
            print('smiles ', count, smile)
            x_size, features, edge_index, atoms = smile_to_graph(smile)
            # make the graph ready for PyTorch Geometrics GAT algorithms:

            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0))
            GCNData.__setitem__('x_size', torch.LongTensor([x_size]))
            GCNData.__setitem__('edge_size', torch.LongTensor([len(edge_index)]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')

        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])