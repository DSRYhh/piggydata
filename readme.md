## Homework 2: Visualization and data preparation

### Tutorial: How to view a Markdown (*.md) file
In the directory of markdown file, right-click in empty space, choose **Open with Code** (Visual Studio Code required), open markdown file in left sidebar (EXPLORER), then use shortcut Ctrl + Shift + V to open markdown previewer.

### Launch Jupyter Notebook
1. In start menu, open Anaconda Prompt
2. Switch path to your code directory (e.g., `cd D:\pig`)
3. Launch Jupyter Notebook with command `jupyter notebook`
![launch jupyter](./img/launch_jupyter.png)
4. Open `plot.ipynb` to continue your homework


## Extra Homework Ⅰ: Code pythonic
Python has its own [code style](https://www.python.org/dev/peps/pep-0008/). So-called *Pythonic* code is the code in Python code style. A pythonic code includes 2 parts:

One is format style. For example, two blank lines are required between two method definition:
```python
# correct
def foo():
    pass


def foobar():
    pass
```
```python
# wrong
def foo():
    pass
def foobar():
    pass
```

Thanks to PyCharm, you no longer need to read the boring official documentation in code style. PyCharm offers powerful formatting tools for you to format your code automatically. What you need to do is just press **`Alt + F8`**, then your code will be formatted in standard python style. Remember **`Alt + Enter`** is also your friend when your code is underlined.

The other one is code style. For example, you do something like this in *C-like* programming language when you change the value of two variables `a` and `b`:
```c
int temp = a;
a = b;
b = temp;
```
The pythonic code only needs one line:
```python
b, a = a, b
```

In this extra homework, you are required to use `Alt + Enter` and `Alt + F8` to clean your code when PyCharm underlines your future code. **Try to avoid wave lines in your code.**

## Homework 3: Predict!


<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

In this homework, you are required to **predict the piggy price** on Shanghai Piggy Dataset. 

To simplify the model, **Linear Regression** will be applied to the dataset. Feel free to explore any complex model although I believe you will not. Linear Regression means you want the output $y$ be a linear combination of feature matrix $X$, as expressed below:
$$y = WX$$
where $W$ is a linear mapping you want to predict.

### Homework 3.1: Data transforming and normalization

Almost all algorithms accept the numerical matrix as input. Since there is a *time* column in Shanghai Piggy Dataset, convert it to numerical data is necessary. About time transforming, see Appendix I.

In this part, NumPy array will be used to represent a numerical matrix. There is a brief introduction of NumPy in Appendix II.

Finish `normalize` method in `prediction.py`. Then run check code in `prediction.py`:
```python
if __name__ == '__main__':
    Checker.normalization_check()
```
If everything goes well, you will see 
```
Normalization test passed.
```
in console output.

### Appendix I Time transforming
How to transform a time string to a number? An intuitive approach is converting directly, e.g., `"1970-01-01"` to `19700101`. One obvious problem is, if there is not a zero padding before month and date, e.g., `"1970-1-1"` and `"1969-12-12"`, the transforming result comes to `197011` and `19691212`. However, `19691212` is **greater** than `197011`, which is not good.

Another solution is *[Unix Timestamp](https://en.wikipedia.org/wiki/Timestamp)*, which defined as the number of **seconds** that have elapsed since 00:00:00 Coordinated Universal Time (UTC), Thursday, 1 January 1970. Pandas has built-in method to convert `datetime` to `timestamp`.

There must be many many other solutions. Feel free to choose what you like.

### Appendix II - Package Introduction: NumPy
Matrix occupies a very important position in data analysis. Almost all different data should be converted to a matrix before fed into algorithms. You must remember a powerful matrix processing tools called MATLAB&trade;. However, MATLAB&trade; is not free and its compatibility with Python is not good.

In Python, there is an alternative called [NumPy](http://www.numpy.org/), which is often described as *MATLAB in Python*. 
#### Basic NumPy Tutorial
```python
import numpy as np

# create numpy array with builtin function
a = np.arange(16)
a = a.reshape((4, 4))  # reshape to a square matrix
print(a)

# choose a part
print(a[1:3, 0:2])

# create numpy array from python list
b = np.array([1, 2, 3, 4])
print(b)

# add with broadcast
print(a + b)

# stack arrays
# relative method: np.vstack() np.hstack() np.stack()
# i.e., stack vertically \ stack horizontally
print(np.vstack((a, b)))

```



Related reference:
- [sklearn.linear_model.LinearRegression.fit](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.fit)

## Special Note: About Appendix
You will see some appendix in following homework. They often contain some tutorials about packages you might use in the homework. These contents are **totally optional**, feel free to ignore them. 

## Homework 3.2: Predict!

In this part, you're required to use the *Linear Regression* model to fit the Shanghai Piggy Dataset.

Finish `train()` method in `prediction.py`. Then run check code:

```python
if __name__ == '__main__':
    Checker.predict_check()
```
If everything goes well, you will see
```
Training test passed.
```
and linear regression coefficient.
 
### Appendix I - Package Introduction: scikit-learn

The tool will be used for prediction is [`scikit-learn`](http://scikit-learn.org/stable/index.html), or `sklearn`. 

Here is an [example](http://scikit-learn.org/stable/modules/linear_model.html#ridge-regression) to fit a linear function $f(x)=\frac{1}{2}x$

```python
from sklearn import linear_model
import numpy as np

reg = linear_model.Ridge()
x = np.arange(1e3)
y = 0.5 * x
x = x.reshape(-1, 1) # the parameter x of the fit method should be an 2-d array
reg.fit(x, y)
print(reg.coef_)
# expected output:
# [ 0.49999999]

```