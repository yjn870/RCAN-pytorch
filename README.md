# RCAN

This repository is implementation of the "Image Super-Resolution Using Very Deep Residual Channel Attention Networks".

<center><img src="./figs/fig2.png"></center>
<center><img src="./figs/fig3.png"></center>
<center><img src="./figs/fig4.png"></center>

## Requirements
- PyTorch
- Tensorflow
- tqdm
- Numpy
- Pillow

**Tensorflow** is required for quickly fetching image in training phase.

## Results

For below results, we set the number of residual groups as 6, the number of RCAB as 12, the number of features as 64. <br />
In addition, we use a intermediate weights because training process need to take a looong time on my computer. 😭<br />

<table>
    <tr>
        <td><center>Original</center></td>
        <td><center>BICUBIC x2</center></td>
        <td><center>RCAN x2</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="./data/monarch.bmp" height="300"></center>
    	</td>
    	<td>
    		<center><img src="./data/monarch_x2_bicubic.png" height="300"></center>
    	</td>
    	<td>
    		<center><img src="./data/monarch_x2_RCAN.png" height="300"></center>
    	</td>
    </tr>
</table>

## Usages

### Train

When training begins, the model weights will be saved every epoch. <br />
If you want to train quickly, you should use **--use_fast_loader** option.

```bash
python main.py --scale 2 \
               --num_rg 10 \
               --num_rcab 20 \ 
               --num_features 64 \              
               --images_dir "" \
               --outputs_dir "" \               
               --patch_size 48 \
               --batch_size 16 \
               --num_epochs 20 \
               --lr 1e-4 \
               --threads 8 \
               --seed 123 \
               --use_fast_loader              
```

### Test

Output results consist of restored images by the BICUBIC and the RCAN.

```bash
python example --scale 2 \
               --num_rg 10 \
               --num_rcab 20 \
               --num_features 64 \
               --weights_path "" \
               --image_path "" \
               --outputs_dir ""                              
```
