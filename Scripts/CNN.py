### Classify hyperspectral images using a CNN
from pathlib import PosixPath
import numpy as np
import random
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class explore_data:
    def show_input(self, n_show=2):
        '''
        Randomly select a few images from both categories and display 
        '''
        npy_paths = random.sample(true_files, n_show) + random.sample(false_files, n_show)
        fig = make_subplots(rows=1, cols=len(npy_paths), subplot_titles=[f'{path.name}<br>{path.parent.name}' for path in npy_paths])
        for i, npy_path in enumerate(npy_paths):
            npy = np.load(npy_path,allow_pickle=True).item()
            img = npy['hyperspectral_data'][:,:,[60,30,20]]
            img = (255 * (img - img.min()) / (np.ptp(img) + 1e-8)).astype(np.uint8)  # Normalize and convert to uint8
            fig.add_trace(go.Image(z=img), row=1, col=i+1,)
        fig.update_layout({'title':f'{n_show*2} Sample Images'})
        fig.show()

        
if __name__ == '__main__':
    true_path = PosixPath('/Users/maycaj/Documents/HSI/Raw Data/npyHCubesTrueCrops')
    false_path = PosixPath('/Users/maycaj/Documents/HSI/Raw Data/npyHCubesFalseCrops')
    true_files = [file for file in true_path.iterdir() if file.is_file()]
    false_files = [file for file in false_path.iterdir() if file.is_file()]
    explorer = explore_data()
    explorer.show_input()
    print('All done')
