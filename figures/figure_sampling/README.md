# The folder creates sampling patterns

### plot 2-shot kx-ky sampling pattern

```bash
python plot_sample_pattern.py --plot_kxky --seg_idx 1
python plot_sample_pattern.py --plot_kxky --seg_idx 2
```

<p align="center">
  <img alt="Light" src="sampling_pattern_kxky_Seg1of2_Pat3.png" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Light" src="sampling_pattern_kxky_Seg2of2_Pat3.png" width="45%">
</p>

### plot 2-shot ky-diff sampling pattern

```bash
python plot_sample_pattern.py
```

<p align="center">
  <img alt="Light" src="sampling_pattern_ky_t_Seg2_Pat3.png" width="45%">
</p>


### plot 1-shot ky-diff sampling pattern

```bash
python plot_sample_pattern.py --seg 1
```

<p align="center">
  <img alt="Light" src="sampling_pattern_ky_t_Seg1_Pat3.png" width="45%">
</p>

### plot multi-shell sampling spheres

```bash
python plot_sphere.py
```

<p align="center">
  <img alt="Light" src="spheres.png" width="45%">
</p>

### Locally Low Rank

```bash
python demo_llr.py
```

<p align="center">
  <img alt="Light" src="DWI.png" width="55%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Light" src="DWI_LLR_Property.png" width="40%">
</p>