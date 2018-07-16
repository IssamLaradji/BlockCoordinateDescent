# Let’s Make Block Coordinate Descent Go Fast

## [Link to the arXiv paper](https://arxiv.org/pdf/1712.08859.pdf)


Reproducing the experiments (Figures 4-13) in the paper
-------------------------------------------------------

### You can run the experiments as follows.

1. Open Terminal (or Command Prompt in Windows);

2. Download this repository by running the command,
```
git clone https://github.com/IssamLaradji/BlockCoordinateDescent.git
```

3. Go into the code's directory,
```
cd BlockCoordinateDescent
```

4. Run the command for a figure number as shown in the sections below.

Below we show how to generate each of the 4-13 Figures in the paper.

###  Figure 4


```python
python main.py -e fig4
```



|         |            |   |
| ------------- |:-------------:| -----:|
| ![](Figures/png_subplots/fig4/0_0.png)     | ![](Figures/png_subplots/fig4/1_0.png) | ![](Figures/png_subplots/fig4/2_0.png) |

### Figure 5


```python
python main.py -e fig5
```



|         |            |   |
| ------------- |:-------------:| -----:|
| ![](Figures/png_subplots/fig5/0_0.png)     | ![](Figures/png_subplots/fig5/1_0.png) | ![](Figures/png_subplots/fig5/2_0.png) |

### Figure 6


```python
python main.py -e fig6a fig6b
```



|         |            |   |
| ------------- |:-------------:| -----:|
| ![](Figures/png_plots/fig6a.png)     | ![](Figures/png_plots/fig6b.png) 

### Figure 7


```python
python main.py -e fig7
```


|         |            |   |
| ------------- |:-------------:| -----:|
| ![](Figures/png_subplots/fig7/0_0.png)     | ![](Figures/png_subplots/fig7/0_1.png) | ![](Figures/png_subplots/fig7/0_2.png) |


### Figure 8


```python
python main.py -e fig8
```


|         |            |   |
| ------------- |:-------------:| -----:|
| ![](Figures/png_plots/fig8.png)


### Figure 9


```python
python main.py -e fig9
```


|         |            |   |
| ------------- |:-------------:| -----:|
| ![](Figures/png_plots/fig9.png)


### Figure 10



```python
python main.py -e fig10
```


|         |            |   |
| ------------- |:-------------:| -----:|
| ![](Figures/png_plots/fig10.png)

### Figure 11



```python
python main.py -e fig11
```


|         |            |   |
| ------------- |:-------------:| -----:|
| ![](Figures/png_plots/fig11.png)


### Figure 12



```python
python main.py -e fig12
```


|         |            |   |
| ------------- |:-------------:| -----:|
| ![](Figures/png_plots/fig12.png)

### Figure 13


```python
python main.py -e fig13
```


|         |            |   |
| ------------- |:-------------:| -----:|
| ![](Figures/png_plots/fig13.png)


## Citation 
If you find the code useful for your research, please cite:

```bibtex
@Article{nutini2017let,
  title={Let's Make Block Coordinate Descent Go Fast: Faster Greedy Rules, Message-Passing, Active-Set Complexity, and Superlinear Convergence},
  author={Nutini, Julie and Laradji, Issam and Schmidt, Mark},
  journal={arXiv},
  year={2017}
}
```
