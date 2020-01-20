from torch_jtnn.mol_tree import MolTree
from multiprocessing import Pool
from argparse import ArgumentParser
from pathlib import Path

class Main():
    def __init__(self,smiles_path,output_path):
        self.smiles_path = Path(smiles_path)
        self.output_path = Path(output_path)

    def __call__(self):

        assert self.smiles_path.is_file() is True ,"{} is not file.".format(self.smiles_path.name)

        with self.smiles_path.open('r') as f:
            smiles_list = [one.strip() for one in f.readlines() if one != '\n']
        print("number of smiles: {}".format(len(smiles_list)))
        self.cset = set()

        with Pool() as p:
            r = p.map_async(self.get_vocab,smiles_list,callback=self.callback)
            while not r.ready():
                print("Now Processing")
                r.wait(timeout=60)

            if r.successful() == False:
                print("Processing Error")
                print(r.get())
        print("Number of Vocab: {}".format(len(self.cset)))
        with self.output_path.open("w") as f:
            for one in self.cset:
                f.write(one+'\n')

    def get_vocab(self,smiles):
        cset = set()
        try:
            mol = MolTree(smiles)
        except Exception as e:
            pass
        else:
            for c in mol.nodes:
                cset.add(c.smiles)
        return cset

    def callback(self,one_set):
        for one in one_set:
            self.cset |= one


def main():
    parser = ArgumentParser()
    parser.add_argument("smiles_path")
    parser.add_argument("output_path")

    args = parser.parse_args()

    proc = Main(**args.__dict__)
    proc()


if __name__=="__main__":
    main()