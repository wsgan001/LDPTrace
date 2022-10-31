# LDPTrace

<div align=center>
<img src=./fig/LDPTrace.png width="80%" ></img>
</div>

This is our Pytorch implementation for the paper:

> Yuntao Du, Yujia Hu, Zhikun Zhang, Ziquan Fang, Lu Chen, Baihua Zheng and Yunjun Gao (2022). LDPTrace: Locally Differentially Private Trajectory Synthesis

## Environment Requirements

- Ubuntu OS
- Python >= 3.8 (Anaconda3 is recommended)

## Dataset

### Oldenburg

1. For Oldenburg dataset, please refer to http://iapg.jade-hs.de/personen/brinkhoff/generator/ to generate the synthesized dataset. The setting parameters we used are as follows:

   + obj./time 0 0
   + maximum time: 1000
   + classes: 1 0
   + max. speed div: 50

2. After obtaining the raw dataset, it needs to be transformed to the standard input format:

   ```
   #0:
   >0:x0,y0;x1,y1;...
   #1:
   >0:x0,y0;x1,y1;...
   #2:
   >0:...
   ...
   ```

   '>0' is a fixed string denoting the start of a trajectory.

   Different format can also work if the type of variable `db` in the code is guaranteed to be `List[Tuple[float, float]]`.

### Porto

The preprocessed Porto dataset is in `data/porto.xz`

## Run

Here's an example of running LDPTrace:

```python
python main.py --dataset oldenburg --grid_num 6 --max_len 0.9 --epsilon 1.0 --re_syn
```

