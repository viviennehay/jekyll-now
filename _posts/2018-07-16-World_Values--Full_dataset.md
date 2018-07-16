---
title:  img src="https://pbs.twimg.com/profile_images/587949417577066499/3uCD4xxY_400x400.jpg" style="float: left; margin: 20px; height: 55px" World Values Analysis
date: 2017-07-16
published: true
---

<br> <br>


## 1. Question definition <br>

This project focuses on understanding the differences between attitudes to work around the world, and what drives those differences. Specifically:  <br> <br>
a) What different values systems exist around the globe (focused on attitudes to work)? <br>
b) What socio-economic factors influence a country's values system? <br>
c) How well can we predict a country's values system, based on these factors?

## 2. Get data

### Import necessary modules & set environment variables


```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from data_dictionary import *
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', 500)

_random_state = 42
```

### Import data


```python
df = pd.read_stata('./world-values-survey-data/WVS_Longitudinal_1981_2014_stata_v2015_04_18.dta', convert_categoricals=False) # index_col='S025', 
```

### Clean data

#### Check shape and nulls; inspect header


```python
df.shape
```




    (341271, 1410)




```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>S001</th>
      <th>S002</th>
      <th>S002EVS</th>
      <th>S003</th>
      <th>S003A</th>
      <th>S004</th>
      <th>S006</th>
      <th>S007</th>
      <th>S007_01</th>
      <th>S008</th>
      <th>S009</th>
      <th>S009A</th>
      <th>S010</th>
      <th>S010_01</th>
      <th>S010_02</th>
      <th>S010_03</th>
      <th>S010_04</th>
      <th>S011</th>
      <th>S012</th>
      <th>S013</th>
      <th>S013B</th>
      <th>S014</th>
      <th>S015</th>
      <th>S016</th>
      <th>S017</th>
      <th>S017A</th>
      <th>S018</th>
      <th>S018A</th>
      <th>S019</th>
      <th>S019A</th>
      <th>S020</th>
      <th>S021</th>
      <th>S021A</th>
      <th>S022</th>
      <th>S023</th>
      <th>S024</th>
      <th>S024A</th>
      <th>S025</th>
      <th>S025A</th>
      <th>S026</th>
      <th>S027</th>
      <th>S028</th>
      <th>A001</th>
      <th>A001_CO</th>
      <th>A002</th>
      <th>A002_CO</th>
      <th>A003</th>
      <th>A003_CO</th>
      <th>A004</th>
      <th>A004_CO</th>
      <th>A005</th>
      <th>A005_CO</th>
      <th>A006</th>
      <th>A006_CO</th>
      <th>A007</th>
      <th>A008</th>
      <th>A009</th>
      <th>A010</th>
      <th>A011</th>
      <th>A012</th>
      <th>A013</th>
      <th>A014</th>
      <th>A015</th>
      <th>A016</th>
      <th>A017</th>
      <th>A018</th>
      <th>A019</th>
      <th>A020</th>
      <th>A021</th>
      <th>A022</th>
      <th>A023</th>
      <th>A024</th>
      <th>A025</th>
      <th>A026</th>
      <th>A026_01</th>
      <th>A027</th>
      <th>A028</th>
      <th>A029</th>
      <th>A030</th>
      <th>A031</th>
      <th>A032</th>
      <th>A033</th>
      <th>A034</th>
      <th>A035</th>
      <th>A036</th>
      <th>A037</th>
      <th>A038</th>
      <th>A039</th>
      <th>A040</th>
      <th>A041</th>
      <th>A042</th>
      <th>A043</th>
      <th>A043_01</th>
      <th>A043_01F</th>
      <th>A043_F</th>
      <th>A043B</th>
      <th>A044</th>
      <th>A045</th>
      <th>A046</th>
      <th>A047</th>
      <th>A048</th>
      <th>A049</th>
      <th>A050</th>
      <th>A050_01</th>
      <th>A050_02</th>
      <th>A050_03</th>
      <th>A050_04</th>
      <th>A051</th>
      <th>A052</th>
      <th>A053</th>
      <th>A054</th>
      <th>A055</th>
      <th>A056</th>
      <th>A057</th>
      <th>A058</th>
      <th>A059</th>
      <th>A060</th>
      <th>A061</th>
      <th>A062</th>
      <th>A063</th>
      <th>A064</th>
      <th>A065</th>
      <th>A066</th>
      <th>A067</th>
      <th>A068</th>
      <th>A069</th>
      <th>A070</th>
      <th>A071</th>
      <th>A071B</th>
      <th>A071C</th>
      <th>A072</th>
      <th>A073</th>
      <th>A074</th>
      <th>A075</th>
      <th>A076</th>
      <th>A077</th>
      <th>A078</th>
      <th>A079</th>
      <th>A080</th>
      <th>A080_F</th>
      <th>A081</th>
      <th>A082</th>
      <th>A083</th>
      <th>A084</th>
      <th>A085</th>
      <th>A086</th>
      <th>A087</th>
      <th>A088</th>
      <th>A088B</th>
      <th>A088C</th>
      <th>A089</th>
      <th>A090</th>
      <th>A091</th>
      <th>A092</th>
      <th>A093</th>
      <th>A094</th>
      <th>A095</th>
      <th>A096</th>
      <th>A097</th>
      <th>A097_F</th>
      <th>A098</th>
      <th>A099</th>
      <th>A100</th>
      <th>A101</th>
      <th>A102</th>
      <th>A103</th>
      <th>A104</th>
      <th>A105</th>
      <th>A106</th>
      <th>A106B</th>
      <th>A106C</th>
      <th>A107</th>
      <th>A108</th>
      <th>A109</th>
      <th>A110</th>
      <th>A111</th>
      <th>A112</th>
      <th>A113</th>
      <th>A114</th>
      <th>A115</th>
      <th>A116</th>
      <th>A117</th>
      <th>A118</th>
      <th>A119</th>
      <th>A120</th>
      <th>A121</th>
      <th>A122</th>
      <th>A123</th>
      <th>A124_01</th>
      <th>A124_02</th>
      <th>A124_03</th>
      <th>A124_04</th>
      <th>A124_05</th>
      <th>A124_06</th>
      <th>A124_07</th>
      <th>A124_08</th>
      <th>A124_09</th>
      <th>A124_10</th>
      <th>A124_11</th>
      <th>A124_12</th>
      <th>A124_13</th>
      <th>A124_14</th>
      <th>A124_15</th>
      <th>A124_16</th>
      <th>A124_17</th>
      <th>A124_18</th>
      <th>A124_19</th>
      <th>A124_20</th>
      <th>A124_21</th>
      <th>A124_22</th>
      <th>A124_23</th>
      <th>A124_24</th>
      <th>A124_25</th>
      <th>A124_26</th>
      <th>A124_27</th>
      <th>A124_28</th>
      <th>A124_29</th>
      <th>A124_30</th>
      <th>A124_31</th>
      <th>A124_32</th>
      <th>A124_33</th>
      <th>A124_34</th>
      <th>A124_35</th>
      <th>A124_36</th>
      <th>A124_37</th>
      <th>A124_38</th>
      <th>A124_39</th>
      <th>A124_40</th>
      <th>A124_41</th>
      <th>A124_42</th>
      <th>A124_43</th>
      <th>A124_44</th>
      <th>A124_45</th>
      <th>A124_46</th>
      <th>A124_47</th>
      <th>A124_48</th>
      <th>A124_49</th>
      <th>A124_50</th>
      <th>A124_51</th>
      <th>A124_52</th>
      <th>A124_53</th>
      <th>A124_54</th>
      <th>A124_55</th>
      <th>A124_56</th>
      <th>A124_57</th>
      <th>A124_58</th>
      <th>A124_59</th>
      <th>A124_60</th>
      <th>A124_61</th>
      <th>A165</th>
      <th>...</th>
      <th>G032</th>
      <th>G033</th>
      <th>G034</th>
      <th>G035</th>
      <th>G036</th>
      <th>G037</th>
      <th>G038</th>
      <th>G039</th>
      <th>G040</th>
      <th>G041</th>
      <th>G042</th>
      <th>G043</th>
      <th>G044</th>
      <th>G045</th>
      <th>G046</th>
      <th>G047</th>
      <th>G048</th>
      <th>G049</th>
      <th>G050</th>
      <th>G051</th>
      <th>H001</th>
      <th>H002_01</th>
      <th>H002_02</th>
      <th>H002_03</th>
      <th>H002_04</th>
      <th>H002_05</th>
      <th>H003_01</th>
      <th>H003_02</th>
      <th>H003_03</th>
      <th>H004</th>
      <th>H005</th>
      <th>H006_01</th>
      <th>H006_02</th>
      <th>H006_03</th>
      <th>H006_04</th>
      <th>H006_05</th>
      <th>H006_06</th>
      <th>H007</th>
      <th>H008_01</th>
      <th>H008_02</th>
      <th>H008_03</th>
      <th>H008_04</th>
      <th>I001</th>
      <th>I002</th>
      <th>U001A</th>
      <th>U001B</th>
      <th>U002A</th>
      <th>U002B</th>
      <th>U003A</th>
      <th>U003B</th>
      <th>U004A</th>
      <th>U004B</th>
      <th>U005A</th>
      <th>U005B</th>
      <th>U006A</th>
      <th>U006B</th>
      <th>V001</th>
      <th>V001A</th>
      <th>V002</th>
      <th>V002A</th>
      <th>V003</th>
      <th>V004A</th>
      <th>V004B</th>
      <th>V004C</th>
      <th>V004D</th>
      <th>V004E</th>
      <th>V004R</th>
      <th>V005</th>
      <th>V006</th>
      <th>V006_2</th>
      <th>V006_3</th>
      <th>V006_4</th>
      <th>V007A</th>
      <th>V007B</th>
      <th>V007C</th>
      <th>V007D</th>
      <th>V008</th>
      <th>V009</th>
      <th>V010</th>
      <th>V011</th>
      <th>V012</th>
      <th>V013</th>
      <th>V014</th>
      <th>V015</th>
      <th>V016</th>
      <th>V017</th>
      <th>V018</th>
      <th>W001</th>
      <th>W001A</th>
      <th>W002A</th>
      <th>W002B</th>
      <th>W002C</th>
      <th>W002D</th>
      <th>W002E</th>
      <th>W002R</th>
      <th>W003</th>
      <th>W004</th>
      <th>W005</th>
      <th>W005_2</th>
      <th>W005_3</th>
      <th>W005_4</th>
      <th>W006A</th>
      <th>W006B</th>
      <th>W006C</th>
      <th>W006D</th>
      <th>W007</th>
      <th>W008</th>
      <th>W009</th>
      <th>W010</th>
      <th>W011</th>
      <th>X001</th>
      <th>X002</th>
      <th>X002_01</th>
      <th>X002_01A</th>
      <th>X002_02</th>
      <th>X002_02A</th>
      <th>X002_03</th>
      <th>X003</th>
      <th>X003R</th>
      <th>X003R2</th>
      <th>X004</th>
      <th>X005</th>
      <th>X006</th>
      <th>X006_01</th>
      <th>X006_02</th>
      <th>X007</th>
      <th>X007_01</th>
      <th>X007_02</th>
      <th>X008</th>
      <th>X009</th>
      <th>X009_01</th>
      <th>X010</th>
      <th>X011</th>
      <th>X011_01</th>
      <th>X011_02</th>
      <th>X011A</th>
      <th>X012</th>
      <th>X013</th>
      <th>X014</th>
      <th>X015</th>
      <th>X016</th>
      <th>X017</th>
      <th>X018</th>
      <th>X019</th>
      <th>X020</th>
      <th>X021</th>
      <th>X022</th>
      <th>X022_01</th>
      <th>X022_02A</th>
      <th>X022_02B</th>
      <th>X022_03A</th>
      <th>X022_03B</th>
      <th>X022_04A</th>
      <th>X022_04B</th>
      <th>X022_05A</th>
      <th>X022_05B</th>
      <th>X022_06A</th>
      <th>X022_06B</th>
      <th>X023</th>
      <th>X023R</th>
      <th>X024</th>
      <th>X024B</th>
      <th>X025</th>
      <th>X025A</th>
      <th>X025B</th>
      <th>X025C</th>
      <th>X025CS</th>
      <th>X025CSWVS</th>
      <th>X025LIT</th>
      <th>X025R</th>
      <th>X026</th>
      <th>X027</th>
      <th>X028</th>
      <th>X028_01</th>
      <th>X029</th>
      <th>X030</th>
      <th>X031</th>
      <th>X032</th>
      <th>X032R</th>
      <th>X032R_01</th>
      <th>X033</th>
      <th>X033R</th>
      <th>X034</th>
      <th>X034R</th>
      <th>X034R_01</th>
      <th>X035_2</th>
      <th>X035_3</th>
      <th>X035_4</th>
      <th>X036</th>
      <th>X036A</th>
      <th>X036B</th>
      <th>X036C</th>
      <th>X036D</th>
      <th>X037</th>
      <th>X037_01</th>
      <th>X037_02</th>
      <th>X038</th>
      <th>X039</th>
      <th>X040</th>
      <th>X041</th>
      <th>X042_2</th>
      <th>X042_3</th>
      <th>X042_4</th>
      <th>X043</th>
      <th>X044</th>
      <th>X045</th>
      <th>X045B</th>
      <th>X046</th>
      <th>X047</th>
      <th>X047A</th>
      <th>X047A_01</th>
      <th>X047B</th>
      <th>X047B_01</th>
      <th>X047C</th>
      <th>X047C_01</th>
      <th>X047CS</th>
      <th>X047D</th>
      <th>X047R</th>
      <th>X048</th>
      <th>X048A</th>
      <th>X048B</th>
      <th>X048C</th>
      <th>X048D</th>
      <th>X048E</th>
      <th>X048F</th>
      <th>X048G</th>
      <th>X048WVS</th>
      <th>X049</th>
      <th>X049CS</th>
      <th>X050</th>
      <th>X051</th>
      <th>X052</th>
      <th>X053</th>
      <th>X054</th>
      <th>X055</th>
      <th>Y001</th>
      <th>Y002</th>
      <th>Y003</th>
      <th>Y010</th>
      <th>Y011</th>
      <th>Y012</th>
      <th>Y013</th>
      <th>Y014</th>
      <th>Y020</th>
      <th>Y021</th>
      <th>Y022</th>
      <th>Y023</th>
      <th>Y024</th>
      <th>TRADRAT5</th>
      <th>survself</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>1</td>
      <td>-4</td>
      <td>392</td>
      <td>392</td>
      <td>-4.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>3.920120e+09</td>
      <td>-4</td>
      <td>JP</td>
      <td>JP</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1.089722</td>
      <td>1.089722</td>
      <td>0.905084</td>
      <td>0.905084</td>
      <td>1.357627</td>
      <td>1.357627</td>
      <td>1981</td>
      <td>3.920121e+10</td>
      <td>3.920121e+10</td>
      <td>-4</td>
      <td>-4</td>
      <td>3921</td>
      <td>3921</td>
      <td>3921981</td>
      <td>3921981</td>
      <td>20150418</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>-2</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-2</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>1</td>
      <td>0</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>...</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>1933</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>49</td>
      <td>4</td>
      <td>2</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>17.0</td>
      <td>6.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-2</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>25</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>7</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>392004</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>392001</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>NaN</td>
      <td>2</td>
      <td>-1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.901</td>
      <td>NaN</td>
      <td>0.553333</td>
      <td>NaN</td>
      <td>0.666667</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.35248</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>-4</td>
      <td>392</td>
      <td>392</td>
      <td>-4.0</td>
      <td>2.0</td>
      <td>2</td>
      <td>3.920120e+09</td>
      <td>-4</td>
      <td>JP</td>
      <td>JP</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1.089722</td>
      <td>1.089722</td>
      <td>0.905084</td>
      <td>0.905084</td>
      <td>1.357627</td>
      <td>1.357627</td>
      <td>1981</td>
      <td>3.920121e+10</td>
      <td>3.920121e+10</td>
      <td>-4</td>
      <td>-4</td>
      <td>3921</td>
      <td>3921</td>
      <td>3921981</td>
      <td>3921981</td>
      <td>20150418</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>1</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>1</td>
      <td>0</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>0</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>...</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>1933</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>49</td>
      <td>4</td>
      <td>2</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>14.0</td>
      <td>3.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-2</td>
      <td>-4</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>41</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>25</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-2</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>392004</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>392001</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>NaN</td>
      <td>1</td>
      <td>-1.0</td>
      <td>0.36594</td>
      <td>0.68776</td>
      <td>0.556</td>
      <td>0.0</td>
      <td>0.220000</td>
      <td>0.398303</td>
      <td>0.666667</td>
      <td>NaN</td>
      <td>0.259259</td>
      <td>0.13600</td>
      <td>0.515189</td>
      <td>-0.548588</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>-4</td>
      <td>392</td>
      <td>392</td>
      <td>-4.0</td>
      <td>3.0</td>
      <td>3</td>
      <td>3.920120e+09</td>
      <td>-4</td>
      <td>JP</td>
      <td>JP</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1.089722</td>
      <td>1.089722</td>
      <td>0.905084</td>
      <td>0.905084</td>
      <td>1.357627</td>
      <td>1.357627</td>
      <td>1981</td>
      <td>3.920121e+10</td>
      <td>3.920121e+10</td>
      <td>-4</td>
      <td>-4</td>
      <td>3921</td>
      <td>3921</td>
      <td>3921981</td>
      <td>3921981</td>
      <td>20150418</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>2</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>...</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>1901</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>81</td>
      <td>6</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>15.0</td>
      <td>4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-2</td>
      <td>-4</td>
      <td>7</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>41</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>41</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>392004</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>392001</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>NaN</td>
      <td>1</td>
      <td>2.0</td>
      <td>0.34294</td>
      <td>0.68776</td>
      <td>0.244</td>
      <td>0.0</td>
      <td>0.440000</td>
      <td>0.141896</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.13600</td>
      <td>0.241489</td>
      <td>-1.603452</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
      <td>-4</td>
      <td>392</td>
      <td>392</td>
      <td>-4.0</td>
      <td>4.0</td>
      <td>4</td>
      <td>3.920120e+09</td>
      <td>-4</td>
      <td>JP</td>
      <td>JP</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1.089722</td>
      <td>1.089722</td>
      <td>0.905084</td>
      <td>0.905084</td>
      <td>1.357627</td>
      <td>1.357627</td>
      <td>1981</td>
      <td>3.920121e+10</td>
      <td>3.920121e+10</td>
      <td>-4</td>
      <td>-4</td>
      <td>3921</td>
      <td>3921</td>
      <td>3921981</td>
      <td>3921981</td>
      <td>20150418</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>2</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>...</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>1901</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>81</td>
      <td>6</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>15.0</td>
      <td>4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-2</td>
      <td>-4</td>
      <td>7</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>41</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>41</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>392004</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>392001</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>NaN</td>
      <td>1</td>
      <td>2.0</td>
      <td>0.34294</td>
      <td>0.68776</td>
      <td>0.244</td>
      <td>0.0</td>
      <td>0.440000</td>
      <td>0.141896</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.13600</td>
      <td>0.241489</td>
      <td>-1.603452</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>1</td>
      <td>-4</td>
      <td>392</td>
      <td>392</td>
      <td>-4.0</td>
      <td>5.0</td>
      <td>5</td>
      <td>3.920120e+09</td>
      <td>-4</td>
      <td>JP</td>
      <td>JP</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1.089722</td>
      <td>1.089722</td>
      <td>0.905084</td>
      <td>0.905084</td>
      <td>1.357627</td>
      <td>1.357627</td>
      <td>1981</td>
      <td>3.920121e+10</td>
      <td>3.920121e+10</td>
      <td>-4</td>
      <td>-4</td>
      <td>3921</td>
      <td>3921</td>
      <td>3921981</td>
      <td>3921981</td>
      <td>20150418</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>2</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>1</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>0</td>
      <td>1</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>...</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>1917</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>65</td>
      <td>6</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>5</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>17.0</td>
      <td>6.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-2</td>
      <td>-4</td>
      <td>5</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>41</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>41</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>8</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>392004</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>392001</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>NaN</td>
      <td>1</td>
      <td>-1.0</td>
      <td>0.43269</td>
      <td>0.68776</td>
      <td>0.823</td>
      <td>0.0</td>
      <td>0.220000</td>
      <td>0.298340</td>
      <td>0.333333</td>
      <td>NaN</td>
      <td>0.222222</td>
      <td>0.13600</td>
      <td>0.129725</td>
      <td>-0.377730</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1410 columns</p>
</div>



#### We can relabel the columns using the data dictionary found at  http://www.worldvaluessurvey.org/WVSDocumentationWVL.jsp


```python
new_column_labels = []

for i in df.columns:
    new_column_labels.append(column_map_dictionary.get(i, i))
```


```python
new_column_labels2 = []

for i in new_column_labels:
    new_column_labels2.append(column_map_dictionary.get(i.lower(), i))
```


```python
df.columns = new_column_labels2
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Study</th>
      <th>Wave</th>
      <th>EVS-wave</th>
      <th>Country/region</th>
      <th>Country/regions [with split ups]</th>
      <th>Set</th>
      <th>Original respondent number</th>
      <th>Unified respondent number</th>
      <th>Unified respondent number (EVS/WVS LF)</th>
      <th>Interviewer number</th>
      <th>S009</th>
      <th>S009A</th>
      <th>Total length of interview</th>
      <th>Time of interview: start hour</th>
      <th>Time of interview: start minute</th>
      <th>Time of interview: end hour</th>
      <th>Time of interview: end minute</th>
      <th>Time at the end of interview</th>
      <th>Date interview</th>
      <th>Respondent interested during the interview</th>
      <th>Interview privacy</th>
      <th>Confidence respondent during the interview</th>
      <th>On the whole respondent looked</th>
      <th>Language in which interview was conducted</th>
      <th>Weight</th>
      <th>Weight [with split ups]</th>
      <th>Equilibrated weight-1000</th>
      <th>Equilibrated weight-1000 [with split ups]</th>
      <th>Equilibrated weight-1500</th>
      <th>Equilibrated weight-1500 [with split ups]</th>
      <th>Year survey</th>
      <th>Country - wave - study - set - year</th>
      <th>Country2 - wave - study - set - year</th>
      <th>Year/month of start-fieldwork</th>
      <th>Year/month of end-fieldwork</th>
      <th>Country - wave</th>
      <th>Country - wave [with split ups]</th>
      <th>Country - year</th>
      <th>Country - year [with split ups]</th>
      <th>File version (yyyymmdd)</th>
      <th>split ballot</th>
      <th>split oecd</th>
      <th>Important in life: Family</th>
      <th>Family important</th>
      <th>Important in life: Friends</th>
      <th>Friends important</th>
      <th>Important in life: Leisure time</th>
      <th>Leisure time</th>
      <th>Important in life: Politics</th>
      <th>Politics important</th>
      <th>Important in life: Work</th>
      <th>Work important</th>
      <th>Important in life: Religion</th>
      <th>Religion important</th>
      <th>Service to others important in life</th>
      <th>Feeling of happiness</th>
      <th>State of health (subjective)</th>
      <th>Ever felt very excited or interested</th>
      <th>Ever felt restless</th>
      <th>Ever felt proud because someone complimented you</th>
      <th>Ever felt very lonely or remote from other people</th>
      <th>Ever felt pleased about having accomplished something</th>
      <th>Ever felt bored</th>
      <th>Ever felt on top of the world</th>
      <th>Ever felt depressed or very unhappy</th>
      <th>Ever felt that things were going your way</th>
      <th>Ever felt upset because somebody criticized you</th>
      <th>When you are home, do you feel relaxed</th>
      <th>When you are home, do you feel anxious</th>
      <th>When you are home, do you feel happy</th>
      <th>When you are home, do you feel aggressive</th>
      <th>When you are home, do you feel secure</th>
      <th>Respect and love for parents</th>
      <th>Parents responsibilities to their children</th>
      <th>Children responsibilities to their parents in need at expense of/not sacrifice own well-being</th>
      <th>Important child qualities: good manners</th>
      <th>Important child qualities: politeness and neatness</th>
      <th>Important child qualities: independence</th>
      <th>Important child qualities: hard work</th>
      <th>Important child qualities: honesty</th>
      <th>Important child qualities: feeling of responsibility</th>
      <th>Important child qualities: patience</th>
      <th>Important child qualities: imagination</th>
      <th>Important child qualities: tolerance and respect for other people</th>
      <th>Important child qualities: leadership</th>
      <th>Important child qualities: self-control</th>
      <th>Important child qualities: thrift saving money and things</th>
      <th>Important child qualities: determination perseverance</th>
      <th>Important child qualities: religious faith</th>
      <th>Important child qualities: unselfishness</th>
      <th>Important child qualities: obedience</th>
      <th>Important child qualities: loyalty</th>
      <th>Important child qualities: none</th>
      <th>Flag variable: learn children at home: none</th>
      <th>Flag variable: learn children at home</th>
      <th>Important child qualities: Self-expression</th>
      <th>What children should learn 1</th>
      <th>What children should learn 2</th>
      <th>Abortion when the mothers health is at risk</th>
      <th>Abortion when child physically handicapped</th>
      <th>Abortion when woman not married</th>
      <th>Abortion if not wanting more children</th>
      <th>Way of spending leisure time</th>
      <th>Leisure time: meeting nice people</th>
      <th>Leisure time: relaxing</th>
      <th>Leisure time: doing as I want</th>
      <th>Leisure time: learning something new</th>
      <th>Spend leisure time: alone</th>
      <th>Spend leisure time: with family</th>
      <th>Spend leisure time: with friends</th>
      <th>Spend leisure time: in a lively place</th>
      <th>Spend leisure time: all equally</th>
      <th>Spend leisure time with: don’t know</th>
      <th>Spend time with parents or other relatives</th>
      <th>Spend time with friends</th>
      <th>Spend time with colleagues from work</th>
      <th>Spend time with people at your church, mosque or synagogue</th>
      <th>Spend time with people at sport, culture, communal organization</th>
      <th>How often discusses political matters with friends</th>
      <th>Persuading friends, relatives or fellow workers</th>
      <th>Member: Belong to social welfare service for elderly</th>
      <th>Member: Belong to religious organization</th>
      <th>Member: Belong to education, arts, music or cultural activities</th>
      <th>Member: Belong to labour unions</th>
      <th>Member: Belong to political parties</th>
      <th>Member: Belong to local political actions</th>
      <th>Member: Belong to human rights</th>
      <th>Member: Belong to conservation, the environment, ecology, animal rights</th>
      <th>Member: Belong to conservation, the environment, ecology</th>
      <th>Member: Belong to animal rights</th>
      <th>Member: Belong to professional associations</th>
      <th>Member: Belong to youth work</th>
      <th>Member: Belong to sports or recreation</th>
      <th>Member: Belong to women´s group</th>
      <th>Member: Belong to peace movement</th>
      <th>Member: Belong to organization concerned with health</th>
      <th>Belong: Cconsumer groups</th>
      <th>Member: Belong to other groups</th>
      <th>Member: Belong to none</th>
      <th>Flag variable: do you belong to: none</th>
      <th>Voluntary work: Unpaid work social welfare service for elderly, handicapped or deprived people</th>
      <th>Voluntary work: Unpaid work religious or church organization</th>
      <th>Voluntary work: Unpaid work education, arts, music or cultural activities</th>
      <th>Voluntary work: Unpaid work labour unions</th>
      <th>Voluntary work: Unpaid work political parties or groups</th>
      <th>Voluntary work: Unpaid work local political action groups</th>
      <th>Voluntary work: Unpaid work human rights</th>
      <th>Voluntary work: Unpaid work environment, conservation, animal rights</th>
      <th>Voluntary work: Unpaid work environment, conservation, ecology</th>
      <th>Voluntary work: Unpaid work animal rights</th>
      <th>Voluntary work: Unpaid work professional associations</th>
      <th>Voluntary work: Unpaid work youth work</th>
      <th>Voluntary work: Unpaid work sports or recreation</th>
      <th>Voluntary work: Unpaid work women´s group</th>
      <th>Voluntary work: Unpaid work peace movement</th>
      <th>Voluntary work: Unpaid work organization concerned with health</th>
      <th>Voluntary work: Unpaid work consumer groups</th>
      <th>Voluntary work: Unpaid work other groups</th>
      <th>Voluntary work: Unpaid work none</th>
      <th>Flag variable: do you work unpaid for: none</th>
      <th>Active/Inactive membership of church or religious organization</th>
      <th>Active/Inactive membership of sport or recreation</th>
      <th>Active/Inactive membership of art, music, educational</th>
      <th>Active/Inactive membership of labour unions</th>
      <th>Active/Inactive membership of political party</th>
      <th>Active/Inactive membership of environmental organization</th>
      <th>Active/Inactive membership of professional organization</th>
      <th>Active/Inactive membership of charitable/humanitarian organization</th>
      <th>Active/Inactive membership of any other organization</th>
      <th>Active/Inactive membership: Consumer organization</th>
      <th>Active/Inactive membership: Self-help group, mutual aid group</th>
      <th>Reasons voluntary work: Solidarity with the poor and disadvantaged</th>
      <th>Reasons voluntary work: Compassion for those in need</th>
      <th>Reasons voluntary work: Opportunity to repay something</th>
      <th>Reasons voluntary work: Sense of duty, moral, obligation</th>
      <th>Reasons voluntary work: Identifying with people who suffer</th>
      <th>Reasons voluntary work: Time on my hands</th>
      <th>Reasons voluntary work: Personal satisfaction</th>
      <th>Reasons voluntary work: Religious belief</th>
      <th>Reasons voluntary work: Help disadvantaged people</th>
      <th>Reasons voluntary work: Make a contribution to my local community</th>
      <th>Reasons voluntary work: Bring about social or political change</th>
      <th>Reasons voluntary work: For social reasons</th>
      <th>Reasons voluntary work: Gain new skills and useful experience</th>
      <th>Reasons voluntary work: Did not want to, but could not refuse</th>
      <th>Dislike being with people with different ideas</th>
      <th>Do you ever feel very lonely</th>
      <th>People´s will to help each other today</th>
      <th>Neighbours: People with a criminal record</th>
      <th>Neighbours: People of a different race</th>
      <th>Neighbours: Heavy drinkers</th>
      <th>Neighbours: Emotionally unstable people</th>
      <th>Neighbours: Muslims</th>
      <th>Neighbours: Immigrants/foreign workers</th>
      <th>Neighbours: People who have AIDS</th>
      <th>Neighbours: Drug addicts</th>
      <th>Neighbours: Homosexuals</th>
      <th>Neighbours: Jews</th>
      <th>Neighbours: Evangelists</th>
      <th>Neighbours: People of a different religion</th>
      <th>Neighbours: People of the same religion</th>
      <th>Neighbours: Militant minority</th>
      <th>Neighbours: Zoroastrians</th>
      <th>Neighbours: People not from country of origin</th>
      <th>Neighbours: Gypsies</th>
      <th>Neighbours: Political Extremists</th>
      <th>Neighbours: Trafficants</th>
      <th>Neighbours: Indians or Lebanese</th>
      <th>Neighbours: Chinese or Philippino Chinese</th>
      <th>Neighbours: Spiritists</th>
      <th>Neighbours: Protestants</th>
      <th>Neighbours: Christians</th>
      <th>Neighbours: Witchdoctors and related labels</th>
      <th>Neighbours: Left wing extremists</th>
      <th>Neighbours: Right wing extremists</th>
      <th>Neighbours: People with large families</th>
      <th>Neighbours: Hindus</th>
      <th>Neighbours: North-American persons</th>
      <th>Neighbours: Haitians</th>
      <th>Neighbours: Members of new religious movements</th>
      <th>Neighbours: Jews, Arabs, Asians, gypsies, etc</th>
      <th>Neighbours: Black people</th>
      <th>Neighbours: White people</th>
      <th>Neighbours: Coloured people</th>
      <th>Neighbours: Indians</th>
      <th>Neighbours: Kurds, Esids</th>
      <th>Neighbours: Students</th>
      <th>Neighbours: Unmarried mothers</th>
      <th>Neighbours: Members of minority religious sects or cults</th>
      <th>Neighbours: Unmarried couples living together</th>
      <th>Neighbours: People who speak a different language</th>
      <th>Neighbours: Members of ETA (terrorists)</th>
      <th>Neighbours: Sunnis</th>
      <th>Neighbours: Shia</th>
      <th>Neighbours: French</th>
      <th>Neighbours: British</th>
      <th>Neighbours: Iranian</th>
      <th>Neighbours: Kuwaiti</th>
      <th>Neighbours: Turkish</th>
      <th>Neighbours: Jordanian</th>
      <th>Neighbours: Kildani</th>
      <th>Neighbours: Indigenes; Aborigenes</th>
      <th>Neighbours: Maori</th>
      <th>Neighbours: Pacific Islanders</th>
      <th>Neighbours: Europeans/Pakeha</th>
      <th>Neighbours: Americans</th>
      <th>Neighbours: Chaldean</th>
      <th>Neighbours: Mapuches</th>
      <th>Neighbours: Russians</th>
      <th>Most people can be trusted</th>
      <th>...</th>
      <th>Ethnic diversity</th>
      <th>Important: to have been born in [country]</th>
      <th>Important: to respect [country nationality] political institutions and laws</th>
      <th>Important: to have [country nationality] ancestry</th>
      <th>Important: to be able to speak [country language]</th>
      <th>Important: to have lived in [country] for a long time</th>
      <th>Immigrants take away jobs from [nationality]</th>
      <th>Immigrants undermine countrys cultural life</th>
      <th>Immigrants increase crime problems</th>
      <th>Immigrants are a strain on welfare system</th>
      <th>Immigrants will become a threat to society</th>
      <th>Immigrants maintain own/take over customs</th>
      <th>Immigrants living in your country: feels like a stranger</th>
      <th>Immigrants living in your country: there are too many</th>
      <th>EU fears: loss of social security</th>
      <th>EU fears: lose national identity/culture</th>
      <th>EU fears: own country pays</th>
      <th>EU fears: loss of power</th>
      <th>EU fears: loss of jobs</th>
      <th>European Union enlargement</th>
      <th>Secure in neighborhood</th>
      <th>Frequency in your neighborhood: Robberies</th>
      <th>Frequency in your neighborhood: Alcohol consumed in the streets</th>
      <th>Frequency in your neighborhood: Police or military interfere with people’s private life</th>
      <th>Frequency in your neighborhood: Racist behavior</th>
      <th>Frequency in your neighborhood: Drug sale in streets</th>
      <th>Things done for reasons of security: Didn’t carry much money</th>
      <th>Things done for reasons of security: Preferred not to go out at night</th>
      <th>Things done for reasons of security: Carried a knife, gun or other weapon</th>
      <th>Respondent was victim of a crime during the past year</th>
      <th>Respondent's family was victim of a crime during last year</th>
      <th>Worries: Losing my job or not finding a job</th>
      <th>Worries: Not being able to give one's children a good education</th>
      <th>Worries: A war involving my country</th>
      <th>Worries: A terrorist attack</th>
      <th>Worries: A civil war</th>
      <th>Worries: Government wire-tapping or reading my mail or email</th>
      <th>Under some conditions, war is necessary to obtain justice</th>
      <th>Frequency you/family (last 12 month): Gone without enough food to eat</th>
      <th>Frequency you/family (last 12 month): Felt unsafe from crime in your own home</th>
      <th>Frequency you/family (last 12 month): Gone without needed medicine or treatment that you needed</th>
      <th>Frequency you/family (last 12 month): Gone without a cash income</th>
      <th>One of the bad effects of science is that it breaks down people’s ideas of right and wrong</th>
      <th>It is not important for me to know about science in my daily life</th>
      <th>Experienced: death of own children</th>
      <th>Age experienced: death of own children</th>
      <th>Experienced: divorce of own children</th>
      <th>Age experienced: divorce of own children</th>
      <th>Experienced: divorce of parents</th>
      <th>Age experienced: divorce of parents</th>
      <th>Experienced: divorce of relative</th>
      <th>Age experienced: divorce of relative</th>
      <th>Experienced: death of father</th>
      <th>Age experienced: death of father</th>
      <th>Experienced: death of mother</th>
      <th>Age experienced: death of mother</th>
      <th>Father born in [country]</th>
      <th>Fathers country of birth - ISO 3166-1 code</th>
      <th>Mother born in [country]</th>
      <th>Mothers country of birth - ISO 3166-1 code</th>
      <th>Lived with parents at the age of 14</th>
      <th>Educational level father [mother]: ISCED-code one digit</th>
      <th>Educational level father [mother]: ISCED-code two digits</th>
      <th>Educational level father [mother]: ISCED-code three digits</th>
      <th>Country specific: Educational level father [mother]</th>
      <th>V004E</th>
      <th>V004R</th>
      <th>Father/mother employed at respondents age of 14</th>
      <th>Job profession/industry father/mother (4 digit isco88)</th>
      <th>V006_2</th>
      <th>V006_3</th>
      <th>V006_4</th>
      <th>Occupational status father/mother - SIOPS</th>
      <th>Occupational status father/mother - ISEI</th>
      <th>Occupational status father/mother - European ESeC</th>
      <th>Occupational status father/mother - egp11</th>
      <th>Father/mother had how many employees</th>
      <th>Did father/mother supervise someone</th>
      <th>How many people did she/he supervise</th>
      <th>Mother liked to read books</th>
      <th>Discussed politics with mother</th>
      <th>Mother liked to follow the news</th>
      <th>Parent(s) had problems making ends meet</th>
      <th>Father liked to read books</th>
      <th>Discussed politics with father</th>
      <th>Father liked to follow the news</th>
      <th>Parent(s) had problems replacing broken things</th>
      <th>Partner/spouse born in [country]</th>
      <th>Spouse/partners country of birth - ISO 3166-1code</th>
      <th>Educational level partner: ISCED-code one digit</th>
      <th>Educational level partner: ISCED-code two digits</th>
      <th>Educational level partner: ISCED-code three digits</th>
      <th>Country specific: Education level partner</th>
      <th>W002E</th>
      <th>W002R</th>
      <th>Paid employment/no paid employment spouse/partner</th>
      <th>Employment/self-employment: last job</th>
      <th>Job profession/industry spouse/partner (4 digit isco88)</th>
      <th>W005_2</th>
      <th>W005_3</th>
      <th>W005_4</th>
      <th>Occupational status spouse/partner - SIOPS</th>
      <th>Occupational status spouse/partner - ISEI</th>
      <th>Occupational status spouse/partner - European ESeC</th>
      <th>Occupational status spouse/partner - egp11</th>
      <th>Spouse/partner had/has how many employees</th>
      <th>Does spouse/partner supervise someone</th>
      <th>How many people does she/he supervise</th>
      <th>Spouse/partner experienced unemployment longer than 3 months</th>
      <th>Dependency on social security during last 5 years spouse/partner</th>
      <th>Sex</th>
      <th>Year of birth</th>
      <th>Having [countrys] nationality</th>
      <th>Respondents nationality - ISO 3166-1 code</th>
      <th>Respondent born in [country]</th>
      <th>Respondents country of birth - ISO 3166-1 code</th>
      <th>Year in which respondent came to live in [country]</th>
      <th>Age</th>
      <th>Age recoded</th>
      <th>Age recoded (3 intervals)</th>
      <th>Stable relationship</th>
      <th>Legally married to partner</th>
      <th>Stable relationship before</th>
      <th>Married to this partner or in registered partnership</th>
      <th>Lived with this partner before marriage/registration of partnership</th>
      <th>Marital status</th>
      <th>Lived with partner before marriage</th>
      <th>Living with partner</th>
      <th>Have you been married before</th>
      <th>Been divorced</th>
      <th>End of relationship because of separation or partners death</th>
      <th>Where r lived after married</th>
      <th>How many children do you have</th>
      <th>How many children do you have - deceased children not included</th>
      <th>Year in which firstborn child was born</th>
      <th>Have you had any children</th>
      <th>How many are still living at home</th>
      <th>Number of people in household</th>
      <th>Number of people in household of 18+</th>
      <th>Number of people in household aged 13-17</th>
      <th>Number of people in household aged 5-12</th>
      <th>Number of people in household under age of 5</th>
      <th>Number of people in household aged 16-17</th>
      <th>Number of people in household aged 11-15</th>
      <th>Number of people in household aged 5-10</th>
      <th>Number of people in household aged 1-4</th>
      <th>Number of people in household under age of 1</th>
      <th>Living in household: partner, husband or wife</th>
      <th>Living in household: children</th>
      <th>Living in household: number of children</th>
      <th>Living in household: parents</th>
      <th>Living in household: number of parents</th>
      <th>Living in household: grandparents</th>
      <th>Living in household: number of grandparents</th>
      <th>Living in household: relatives</th>
      <th>Living in household: number of relatives</th>
      <th>Living in household: non relatives</th>
      <th>Living in household: number of non relatives</th>
      <th>What age did you complete your education</th>
      <th>What age did you complete your education (recoded in intervals)</th>
      <th>Had formal education</th>
      <th>Respondent - literate</th>
      <th>Highest educational level attained</th>
      <th>Educational level respondent: ISCED- code one digit</th>
      <th>Educational level respondent: ISCED-code two digits</th>
      <th>Educational level respondent: ISCED-code three digits</th>
      <th>Education (country specific)</th>
      <th>X025CSWVS</th>
      <th>Was the respondent literate</th>
      <th>Education level (recoded)</th>
      <th>Do you live with your parents</th>
      <th>House or apartment</th>
      <th>Employment status</th>
      <th>Employment/self-employment: last job</th>
      <th>Chief wage earner employed now</th>
      <th>Do you own your home or rent it</th>
      <th>Are you supervising someone</th>
      <th>Number of supervised people</th>
      <th>Number of supervised people (recoded)</th>
      <th>Number of supervised people, 3 cat</th>
      <th>Number of others working in the organization</th>
      <th>Number of others working in the organization (recoded)</th>
      <th>Number of employees</th>
      <th>Number of employees (recoded)</th>
      <th>Number of employees, 4 cat</th>
      <th>Job profession/industry (2 digit isco88)</th>
      <th>Job profession/industry (3 digit isco88)</th>
      <th>Job profession/industry (4 digit isco88)</th>
      <th>Profession/job</th>
      <th>Occupational status respondent - SIOPS</th>
      <th>Occupational status respondent - ISEI</th>
      <th>Occupational status respondent - egp11</th>
      <th>Occupational status respondent - European ESeC</th>
      <th>How long unemployed</th>
      <th>Respondent experienced unemployment longer than 3 months</th>
      <th>Dependency on social security during last 5 years respondent</th>
      <th>How many people work in your department-organization</th>
      <th>Do you or your spouse belong to a labour union</th>
      <th>Are you the chief wage earner in your house</th>
      <th>Is the chief wage earner employed now</th>
      <th>Profession/industry (2 digit isco88)</th>
      <th>Profession/industry (3 digit isco88)</th>
      <th>Profession/industry (4 digit isco88)</th>
      <th>Chief wage earner profession/job</th>
      <th>Family savings during past year</th>
      <th>Social class (subjective)</th>
      <th>Social class (subjective) with 6 categories</th>
      <th>Socio-economic status of respondent</th>
      <th>Scale of incomes</th>
      <th>Weekly household income</th>
      <th>Country specific: Weekly household income</th>
      <th>Monthly household income</th>
      <th>Country specific:Monthly household income</th>
      <th>Annual household income</th>
      <th>Country specific: Annual household income</th>
      <th>Income (country specific)</th>
      <th>Monthly household income (x1000), corrected for ppp in euros</th>
      <th>Income level</th>
      <th>Region where the interview was conducted</th>
      <th>Region: NUTS-1 code</th>
      <th>Region: NUTS-2 code</th>
      <th>Region: NUTS-3 code</th>
      <th>Region at age 14: country</th>
      <th>Region at age 14: NUTS-1 code</th>
      <th>Region at age 14: NUTS-2 code</th>
      <th>Region at age 14: NUTS-3 code</th>
      <th>X048WVS</th>
      <th>Size of town</th>
      <th>Size of town (country specific)</th>
      <th>Type of habitat</th>
      <th>Ethnic group</th>
      <th>Institution of occupation</th>
      <th>Nature of tasks: manual vs. Cognitive</th>
      <th>Nature of tasks: routine vs. Creative</th>
      <th>Nature of tasks: independence</th>
      <th>Post-Materialist index 12-item</th>
      <th>Post-Materialist index 4-item</th>
      <th>Autonomy Index</th>
      <th>Y010</th>
      <th>Y011</th>
      <th>Y012</th>
      <th>Y013</th>
      <th>Y014</th>
      <th>Y020</th>
      <th>Y021</th>
      <th>Y022</th>
      <th>Y023</th>
      <th>Y024</th>
      <th>TRADITIONAL/SECULAR RATIONAL VALUES</th>
      <th>SURVIVAL/SELF-EXPRESSION VALUES</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>1</td>
      <td>-4</td>
      <td>392</td>
      <td>392</td>
      <td>-4.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>3.920120e+09</td>
      <td>-4</td>
      <td>JP</td>
      <td>JP</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1.089722</td>
      <td>1.089722</td>
      <td>0.905084</td>
      <td>0.905084</td>
      <td>1.357627</td>
      <td>1.357627</td>
      <td>1981</td>
      <td>3.920121e+10</td>
      <td>3.920121e+10</td>
      <td>-4</td>
      <td>-4</td>
      <td>3921</td>
      <td>3921</td>
      <td>3921981</td>
      <td>3921981</td>
      <td>20150418</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>-2</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-2</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>1</td>
      <td>0</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>...</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>1933</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>49</td>
      <td>4</td>
      <td>2</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>17.0</td>
      <td>6.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-2</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>25</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>7</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>392004</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>392001</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>NaN</td>
      <td>2</td>
      <td>-1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.901</td>
      <td>NaN</td>
      <td>0.553333</td>
      <td>NaN</td>
      <td>0.666667</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.35248</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>-4</td>
      <td>392</td>
      <td>392</td>
      <td>-4.0</td>
      <td>2.0</td>
      <td>2</td>
      <td>3.920120e+09</td>
      <td>-4</td>
      <td>JP</td>
      <td>JP</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1.089722</td>
      <td>1.089722</td>
      <td>0.905084</td>
      <td>0.905084</td>
      <td>1.357627</td>
      <td>1.357627</td>
      <td>1981</td>
      <td>3.920121e+10</td>
      <td>3.920121e+10</td>
      <td>-4</td>
      <td>-4</td>
      <td>3921</td>
      <td>3921</td>
      <td>3921981</td>
      <td>3921981</td>
      <td>20150418</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>1</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>1</td>
      <td>0</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>0</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>...</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>1933</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>49</td>
      <td>4</td>
      <td>2</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>14.0</td>
      <td>3.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-2</td>
      <td>-4</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>41</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>25</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-2</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>392004</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>392001</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>NaN</td>
      <td>1</td>
      <td>-1.0</td>
      <td>0.36594</td>
      <td>0.68776</td>
      <td>0.556</td>
      <td>0.0</td>
      <td>0.220000</td>
      <td>0.398303</td>
      <td>0.666667</td>
      <td>NaN</td>
      <td>0.259259</td>
      <td>0.13600</td>
      <td>0.515189</td>
      <td>-0.548588</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>-4</td>
      <td>392</td>
      <td>392</td>
      <td>-4.0</td>
      <td>3.0</td>
      <td>3</td>
      <td>3.920120e+09</td>
      <td>-4</td>
      <td>JP</td>
      <td>JP</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1.089722</td>
      <td>1.089722</td>
      <td>0.905084</td>
      <td>0.905084</td>
      <td>1.357627</td>
      <td>1.357627</td>
      <td>1981</td>
      <td>3.920121e+10</td>
      <td>3.920121e+10</td>
      <td>-4</td>
      <td>-4</td>
      <td>3921</td>
      <td>3921</td>
      <td>3921981</td>
      <td>3921981</td>
      <td>20150418</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>2</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>...</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>1901</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>81</td>
      <td>6</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>15.0</td>
      <td>4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-2</td>
      <td>-4</td>
      <td>7</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>41</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>41</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>392004</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>392001</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>NaN</td>
      <td>1</td>
      <td>2.0</td>
      <td>0.34294</td>
      <td>0.68776</td>
      <td>0.244</td>
      <td>0.0</td>
      <td>0.440000</td>
      <td>0.141896</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.13600</td>
      <td>0.241489</td>
      <td>-1.603452</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
      <td>-4</td>
      <td>392</td>
      <td>392</td>
      <td>-4.0</td>
      <td>4.0</td>
      <td>4</td>
      <td>3.920120e+09</td>
      <td>-4</td>
      <td>JP</td>
      <td>JP</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1.089722</td>
      <td>1.089722</td>
      <td>0.905084</td>
      <td>0.905084</td>
      <td>1.357627</td>
      <td>1.357627</td>
      <td>1981</td>
      <td>3.920121e+10</td>
      <td>3.920121e+10</td>
      <td>-4</td>
      <td>-4</td>
      <td>3921</td>
      <td>3921</td>
      <td>3921981</td>
      <td>3921981</td>
      <td>20150418</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>2</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>...</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>1901</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>81</td>
      <td>6</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>15.0</td>
      <td>4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-2</td>
      <td>-4</td>
      <td>7</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>41</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>41</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>392004</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>392001</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>NaN</td>
      <td>1</td>
      <td>2.0</td>
      <td>0.34294</td>
      <td>0.68776</td>
      <td>0.244</td>
      <td>0.0</td>
      <td>0.440000</td>
      <td>0.141896</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.13600</td>
      <td>0.241489</td>
      <td>-1.603452</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>1</td>
      <td>-4</td>
      <td>392</td>
      <td>392</td>
      <td>-4.0</td>
      <td>5.0</td>
      <td>5</td>
      <td>3.920120e+09</td>
      <td>-4</td>
      <td>JP</td>
      <td>JP</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1.089722</td>
      <td>1.089722</td>
      <td>0.905084</td>
      <td>0.905084</td>
      <td>1.357627</td>
      <td>1.357627</td>
      <td>1981</td>
      <td>3.920121e+10</td>
      <td>3.920121e+10</td>
      <td>-4</td>
      <td>-4</td>
      <td>3921</td>
      <td>3921</td>
      <td>3921981</td>
      <td>3921981</td>
      <td>20150418</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>2</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>1</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>0</td>
      <td>1</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>...</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>1917</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>65</td>
      <td>6</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>5</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>17.0</td>
      <td>6.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-2</td>
      <td>-4</td>
      <td>5</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>41</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>41</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>8</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>392004</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>392001</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>NaN</td>
      <td>1</td>
      <td>-1.0</td>
      <td>0.43269</td>
      <td>0.68776</td>
      <td>0.823</td>
      <td>0.0</td>
      <td>0.220000</td>
      <td>0.298340</td>
      <td>0.333333</td>
      <td>NaN</td>
      <td>0.222222</td>
      <td>0.13600</td>
      <td>0.129725</td>
      <td>-0.377730</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1410 columns</p>
</div>



#### Note that the country is listed as a code (in 'Country/region' column). We want to add the name of the country


```python
# Turn each country/ code combination into an item in a list, by splitting on new lines
country_list = country_list.split('\n')

# Create a list of country codes and country names
country_code_list = []
country_name_list = []
for country_pair in country_list:
    country_code_list.append(country_pair.split(':')[0])
    country_name_list.append(country_pair.split(':')[1])
    
# Turn the country code / name lookup into a dataframe, to allow a merge with the original dataframe
country_dictionary = {'country_code': country_code_list, 'country_name': country_name_list}
country_lookup = pd.DataFrame(country_dictionary)
```


```python
# Merge country names into original dataframe
df['Country/region'] = df['Country/region'].astype(int) # match dtypes
country_lookup['country_code'] = country_lookup['country_code'].astype(int)
df = pd.merge(df, country_lookup, left_on='Country/region', right_on='country_code', how = 'left')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Study</th>
      <th>Wave</th>
      <th>EVS-wave</th>
      <th>Country/region</th>
      <th>Country/regions [with split ups]</th>
      <th>Set</th>
      <th>Original respondent number</th>
      <th>Unified respondent number</th>
      <th>Unified respondent number (EVS/WVS LF)</th>
      <th>Interviewer number</th>
      <th>S009</th>
      <th>S009A</th>
      <th>Total length of interview</th>
      <th>Time of interview: start hour</th>
      <th>Time of interview: start minute</th>
      <th>Time of interview: end hour</th>
      <th>Time of interview: end minute</th>
      <th>Time at the end of interview</th>
      <th>Date interview</th>
      <th>Respondent interested during the interview</th>
      <th>Interview privacy</th>
      <th>Confidence respondent during the interview</th>
      <th>On the whole respondent looked</th>
      <th>Language in which interview was conducted</th>
      <th>Weight</th>
      <th>Weight [with split ups]</th>
      <th>Equilibrated weight-1000</th>
      <th>Equilibrated weight-1000 [with split ups]</th>
      <th>Equilibrated weight-1500</th>
      <th>Equilibrated weight-1500 [with split ups]</th>
      <th>Year survey</th>
      <th>Country - wave - study - set - year</th>
      <th>Country2 - wave - study - set - year</th>
      <th>Year/month of start-fieldwork</th>
      <th>Year/month of end-fieldwork</th>
      <th>Country - wave</th>
      <th>Country - wave [with split ups]</th>
      <th>Country - year</th>
      <th>Country - year [with split ups]</th>
      <th>File version (yyyymmdd)</th>
      <th>split ballot</th>
      <th>split oecd</th>
      <th>Important in life: Family</th>
      <th>Family important</th>
      <th>Important in life: Friends</th>
      <th>Friends important</th>
      <th>Important in life: Leisure time</th>
      <th>Leisure time</th>
      <th>Important in life: Politics</th>
      <th>Politics important</th>
      <th>Important in life: Work</th>
      <th>Work important</th>
      <th>Important in life: Religion</th>
      <th>Religion important</th>
      <th>Service to others important in life</th>
      <th>Feeling of happiness</th>
      <th>State of health (subjective)</th>
      <th>Ever felt very excited or interested</th>
      <th>Ever felt restless</th>
      <th>Ever felt proud because someone complimented you</th>
      <th>Ever felt very lonely or remote from other people</th>
      <th>Ever felt pleased about having accomplished something</th>
      <th>Ever felt bored</th>
      <th>Ever felt on top of the world</th>
      <th>Ever felt depressed or very unhappy</th>
      <th>Ever felt that things were going your way</th>
      <th>Ever felt upset because somebody criticized you</th>
      <th>When you are home, do you feel relaxed</th>
      <th>When you are home, do you feel anxious</th>
      <th>When you are home, do you feel happy</th>
      <th>When you are home, do you feel aggressive</th>
      <th>When you are home, do you feel secure</th>
      <th>Respect and love for parents</th>
      <th>Parents responsibilities to their children</th>
      <th>Children responsibilities to their parents in need at expense of/not sacrifice own well-being</th>
      <th>Important child qualities: good manners</th>
      <th>Important child qualities: politeness and neatness</th>
      <th>Important child qualities: independence</th>
      <th>Important child qualities: hard work</th>
      <th>Important child qualities: honesty</th>
      <th>Important child qualities: feeling of responsibility</th>
      <th>Important child qualities: patience</th>
      <th>Important child qualities: imagination</th>
      <th>Important child qualities: tolerance and respect for other people</th>
      <th>Important child qualities: leadership</th>
      <th>Important child qualities: self-control</th>
      <th>Important child qualities: thrift saving money and things</th>
      <th>Important child qualities: determination perseverance</th>
      <th>Important child qualities: religious faith</th>
      <th>Important child qualities: unselfishness</th>
      <th>Important child qualities: obedience</th>
      <th>Important child qualities: loyalty</th>
      <th>Important child qualities: none</th>
      <th>Flag variable: learn children at home: none</th>
      <th>Flag variable: learn children at home</th>
      <th>Important child qualities: Self-expression</th>
      <th>What children should learn 1</th>
      <th>What children should learn 2</th>
      <th>Abortion when the mothers health is at risk</th>
      <th>Abortion when child physically handicapped</th>
      <th>Abortion when woman not married</th>
      <th>Abortion if not wanting more children</th>
      <th>Way of spending leisure time</th>
      <th>Leisure time: meeting nice people</th>
      <th>Leisure time: relaxing</th>
      <th>Leisure time: doing as I want</th>
      <th>Leisure time: learning something new</th>
      <th>Spend leisure time: alone</th>
      <th>Spend leisure time: with family</th>
      <th>Spend leisure time: with friends</th>
      <th>Spend leisure time: in a lively place</th>
      <th>Spend leisure time: all equally</th>
      <th>Spend leisure time with: don’t know</th>
      <th>Spend time with parents or other relatives</th>
      <th>Spend time with friends</th>
      <th>Spend time with colleagues from work</th>
      <th>Spend time with people at your church, mosque or synagogue</th>
      <th>Spend time with people at sport, culture, communal organization</th>
      <th>How often discusses political matters with friends</th>
      <th>Persuading friends, relatives or fellow workers</th>
      <th>Member: Belong to social welfare service for elderly</th>
      <th>Member: Belong to religious organization</th>
      <th>Member: Belong to education, arts, music or cultural activities</th>
      <th>Member: Belong to labour unions</th>
      <th>Member: Belong to political parties</th>
      <th>Member: Belong to local political actions</th>
      <th>Member: Belong to human rights</th>
      <th>Member: Belong to conservation, the environment, ecology, animal rights</th>
      <th>Member: Belong to conservation, the environment, ecology</th>
      <th>Member: Belong to animal rights</th>
      <th>Member: Belong to professional associations</th>
      <th>Member: Belong to youth work</th>
      <th>Member: Belong to sports or recreation</th>
      <th>Member: Belong to women´s group</th>
      <th>Member: Belong to peace movement</th>
      <th>Member: Belong to organization concerned with health</th>
      <th>Belong: Cconsumer groups</th>
      <th>Member: Belong to other groups</th>
      <th>Member: Belong to none</th>
      <th>Flag variable: do you belong to: none</th>
      <th>Voluntary work: Unpaid work social welfare service for elderly, handicapped or deprived people</th>
      <th>Voluntary work: Unpaid work religious or church organization</th>
      <th>Voluntary work: Unpaid work education, arts, music or cultural activities</th>
      <th>Voluntary work: Unpaid work labour unions</th>
      <th>Voluntary work: Unpaid work political parties or groups</th>
      <th>Voluntary work: Unpaid work local political action groups</th>
      <th>Voluntary work: Unpaid work human rights</th>
      <th>Voluntary work: Unpaid work environment, conservation, animal rights</th>
      <th>Voluntary work: Unpaid work environment, conservation, ecology</th>
      <th>Voluntary work: Unpaid work animal rights</th>
      <th>Voluntary work: Unpaid work professional associations</th>
      <th>Voluntary work: Unpaid work youth work</th>
      <th>Voluntary work: Unpaid work sports or recreation</th>
      <th>Voluntary work: Unpaid work women´s group</th>
      <th>Voluntary work: Unpaid work peace movement</th>
      <th>Voluntary work: Unpaid work organization concerned with health</th>
      <th>Voluntary work: Unpaid work consumer groups</th>
      <th>Voluntary work: Unpaid work other groups</th>
      <th>Voluntary work: Unpaid work none</th>
      <th>Flag variable: do you work unpaid for: none</th>
      <th>Active/Inactive membership of church or religious organization</th>
      <th>Active/Inactive membership of sport or recreation</th>
      <th>Active/Inactive membership of art, music, educational</th>
      <th>Active/Inactive membership of labour unions</th>
      <th>Active/Inactive membership of political party</th>
      <th>Active/Inactive membership of environmental organization</th>
      <th>Active/Inactive membership of professional organization</th>
      <th>Active/Inactive membership of charitable/humanitarian organization</th>
      <th>Active/Inactive membership of any other organization</th>
      <th>Active/Inactive membership: Consumer organization</th>
      <th>Active/Inactive membership: Self-help group, mutual aid group</th>
      <th>Reasons voluntary work: Solidarity with the poor and disadvantaged</th>
      <th>Reasons voluntary work: Compassion for those in need</th>
      <th>Reasons voluntary work: Opportunity to repay something</th>
      <th>Reasons voluntary work: Sense of duty, moral, obligation</th>
      <th>Reasons voluntary work: Identifying with people who suffer</th>
      <th>Reasons voluntary work: Time on my hands</th>
      <th>Reasons voluntary work: Personal satisfaction</th>
      <th>Reasons voluntary work: Religious belief</th>
      <th>Reasons voluntary work: Help disadvantaged people</th>
      <th>Reasons voluntary work: Make a contribution to my local community</th>
      <th>Reasons voluntary work: Bring about social or political change</th>
      <th>Reasons voluntary work: For social reasons</th>
      <th>Reasons voluntary work: Gain new skills and useful experience</th>
      <th>Reasons voluntary work: Did not want to, but could not refuse</th>
      <th>Dislike being with people with different ideas</th>
      <th>Do you ever feel very lonely</th>
      <th>People´s will to help each other today</th>
      <th>Neighbours: People with a criminal record</th>
      <th>Neighbours: People of a different race</th>
      <th>Neighbours: Heavy drinkers</th>
      <th>Neighbours: Emotionally unstable people</th>
      <th>Neighbours: Muslims</th>
      <th>Neighbours: Immigrants/foreign workers</th>
      <th>Neighbours: People who have AIDS</th>
      <th>Neighbours: Drug addicts</th>
      <th>Neighbours: Homosexuals</th>
      <th>Neighbours: Jews</th>
      <th>Neighbours: Evangelists</th>
      <th>Neighbours: People of a different religion</th>
      <th>Neighbours: People of the same religion</th>
      <th>Neighbours: Militant minority</th>
      <th>Neighbours: Zoroastrians</th>
      <th>Neighbours: People not from country of origin</th>
      <th>Neighbours: Gypsies</th>
      <th>Neighbours: Political Extremists</th>
      <th>Neighbours: Trafficants</th>
      <th>Neighbours: Indians or Lebanese</th>
      <th>Neighbours: Chinese or Philippino Chinese</th>
      <th>Neighbours: Spiritists</th>
      <th>Neighbours: Protestants</th>
      <th>Neighbours: Christians</th>
      <th>Neighbours: Witchdoctors and related labels</th>
      <th>Neighbours: Left wing extremists</th>
      <th>Neighbours: Right wing extremists</th>
      <th>Neighbours: People with large families</th>
      <th>Neighbours: Hindus</th>
      <th>Neighbours: North-American persons</th>
      <th>Neighbours: Haitians</th>
      <th>Neighbours: Members of new religious movements</th>
      <th>Neighbours: Jews, Arabs, Asians, gypsies, etc</th>
      <th>Neighbours: Black people</th>
      <th>Neighbours: White people</th>
      <th>Neighbours: Coloured people</th>
      <th>Neighbours: Indians</th>
      <th>Neighbours: Kurds, Esids</th>
      <th>Neighbours: Students</th>
      <th>Neighbours: Unmarried mothers</th>
      <th>Neighbours: Members of minority religious sects or cults</th>
      <th>Neighbours: Unmarried couples living together</th>
      <th>Neighbours: People who speak a different language</th>
      <th>Neighbours: Members of ETA (terrorists)</th>
      <th>Neighbours: Sunnis</th>
      <th>Neighbours: Shia</th>
      <th>Neighbours: French</th>
      <th>Neighbours: British</th>
      <th>Neighbours: Iranian</th>
      <th>Neighbours: Kuwaiti</th>
      <th>Neighbours: Turkish</th>
      <th>Neighbours: Jordanian</th>
      <th>Neighbours: Kildani</th>
      <th>Neighbours: Indigenes; Aborigenes</th>
      <th>Neighbours: Maori</th>
      <th>Neighbours: Pacific Islanders</th>
      <th>Neighbours: Europeans/Pakeha</th>
      <th>Neighbours: Americans</th>
      <th>Neighbours: Chaldean</th>
      <th>Neighbours: Mapuches</th>
      <th>Neighbours: Russians</th>
      <th>Most people can be trusted</th>
      <th>...</th>
      <th>Important: to respect [country nationality] political institutions and laws</th>
      <th>Important: to have [country nationality] ancestry</th>
      <th>Important: to be able to speak [country language]</th>
      <th>Important: to have lived in [country] for a long time</th>
      <th>Immigrants take away jobs from [nationality]</th>
      <th>Immigrants undermine countrys cultural life</th>
      <th>Immigrants increase crime problems</th>
      <th>Immigrants are a strain on welfare system</th>
      <th>Immigrants will become a threat to society</th>
      <th>Immigrants maintain own/take over customs</th>
      <th>Immigrants living in your country: feels like a stranger</th>
      <th>Immigrants living in your country: there are too many</th>
      <th>EU fears: loss of social security</th>
      <th>EU fears: lose national identity/culture</th>
      <th>EU fears: own country pays</th>
      <th>EU fears: loss of power</th>
      <th>EU fears: loss of jobs</th>
      <th>European Union enlargement</th>
      <th>Secure in neighborhood</th>
      <th>Frequency in your neighborhood: Robberies</th>
      <th>Frequency in your neighborhood: Alcohol consumed in the streets</th>
      <th>Frequency in your neighborhood: Police or military interfere with people’s private life</th>
      <th>Frequency in your neighborhood: Racist behavior</th>
      <th>Frequency in your neighborhood: Drug sale in streets</th>
      <th>Things done for reasons of security: Didn’t carry much money</th>
      <th>Things done for reasons of security: Preferred not to go out at night</th>
      <th>Things done for reasons of security: Carried a knife, gun or other weapon</th>
      <th>Respondent was victim of a crime during the past year</th>
      <th>Respondent's family was victim of a crime during last year</th>
      <th>Worries: Losing my job or not finding a job</th>
      <th>Worries: Not being able to give one's children a good education</th>
      <th>Worries: A war involving my country</th>
      <th>Worries: A terrorist attack</th>
      <th>Worries: A civil war</th>
      <th>Worries: Government wire-tapping or reading my mail or email</th>
      <th>Under some conditions, war is necessary to obtain justice</th>
      <th>Frequency you/family (last 12 month): Gone without enough food to eat</th>
      <th>Frequency you/family (last 12 month): Felt unsafe from crime in your own home</th>
      <th>Frequency you/family (last 12 month): Gone without needed medicine or treatment that you needed</th>
      <th>Frequency you/family (last 12 month): Gone without a cash income</th>
      <th>One of the bad effects of science is that it breaks down people’s ideas of right and wrong</th>
      <th>It is not important for me to know about science in my daily life</th>
      <th>Experienced: death of own children</th>
      <th>Age experienced: death of own children</th>
      <th>Experienced: divorce of own children</th>
      <th>Age experienced: divorce of own children</th>
      <th>Experienced: divorce of parents</th>
      <th>Age experienced: divorce of parents</th>
      <th>Experienced: divorce of relative</th>
      <th>Age experienced: divorce of relative</th>
      <th>Experienced: death of father</th>
      <th>Age experienced: death of father</th>
      <th>Experienced: death of mother</th>
      <th>Age experienced: death of mother</th>
      <th>Father born in [country]</th>
      <th>Fathers country of birth - ISO 3166-1 code</th>
      <th>Mother born in [country]</th>
      <th>Mothers country of birth - ISO 3166-1 code</th>
      <th>Lived with parents at the age of 14</th>
      <th>Educational level father [mother]: ISCED-code one digit</th>
      <th>Educational level father [mother]: ISCED-code two digits</th>
      <th>Educational level father [mother]: ISCED-code three digits</th>
      <th>Country specific: Educational level father [mother]</th>
      <th>V004E</th>
      <th>V004R</th>
      <th>Father/mother employed at respondents age of 14</th>
      <th>Job profession/industry father/mother (4 digit isco88)</th>
      <th>V006_2</th>
      <th>V006_3</th>
      <th>V006_4</th>
      <th>Occupational status father/mother - SIOPS</th>
      <th>Occupational status father/mother - ISEI</th>
      <th>Occupational status father/mother - European ESeC</th>
      <th>Occupational status father/mother - egp11</th>
      <th>Father/mother had how many employees</th>
      <th>Did father/mother supervise someone</th>
      <th>How many people did she/he supervise</th>
      <th>Mother liked to read books</th>
      <th>Discussed politics with mother</th>
      <th>Mother liked to follow the news</th>
      <th>Parent(s) had problems making ends meet</th>
      <th>Father liked to read books</th>
      <th>Discussed politics with father</th>
      <th>Father liked to follow the news</th>
      <th>Parent(s) had problems replacing broken things</th>
      <th>Partner/spouse born in [country]</th>
      <th>Spouse/partners country of birth - ISO 3166-1code</th>
      <th>Educational level partner: ISCED-code one digit</th>
      <th>Educational level partner: ISCED-code two digits</th>
      <th>Educational level partner: ISCED-code three digits</th>
      <th>Country specific: Education level partner</th>
      <th>W002E</th>
      <th>W002R</th>
      <th>Paid employment/no paid employment spouse/partner</th>
      <th>Employment/self-employment: last job</th>
      <th>Job profession/industry spouse/partner (4 digit isco88)</th>
      <th>W005_2</th>
      <th>W005_3</th>
      <th>W005_4</th>
      <th>Occupational status spouse/partner - SIOPS</th>
      <th>Occupational status spouse/partner - ISEI</th>
      <th>Occupational status spouse/partner - European ESeC</th>
      <th>Occupational status spouse/partner - egp11</th>
      <th>Spouse/partner had/has how many employees</th>
      <th>Does spouse/partner supervise someone</th>
      <th>How many people does she/he supervise</th>
      <th>Spouse/partner experienced unemployment longer than 3 months</th>
      <th>Dependency on social security during last 5 years spouse/partner</th>
      <th>Sex</th>
      <th>Year of birth</th>
      <th>Having [countrys] nationality</th>
      <th>Respondents nationality - ISO 3166-1 code</th>
      <th>Respondent born in [country]</th>
      <th>Respondents country of birth - ISO 3166-1 code</th>
      <th>Year in which respondent came to live in [country]</th>
      <th>Age</th>
      <th>Age recoded</th>
      <th>Age recoded (3 intervals)</th>
      <th>Stable relationship</th>
      <th>Legally married to partner</th>
      <th>Stable relationship before</th>
      <th>Married to this partner or in registered partnership</th>
      <th>Lived with this partner before marriage/registration of partnership</th>
      <th>Marital status</th>
      <th>Lived with partner before marriage</th>
      <th>Living with partner</th>
      <th>Have you been married before</th>
      <th>Been divorced</th>
      <th>End of relationship because of separation or partners death</th>
      <th>Where r lived after married</th>
      <th>How many children do you have</th>
      <th>How many children do you have - deceased children not included</th>
      <th>Year in which firstborn child was born</th>
      <th>Have you had any children</th>
      <th>How many are still living at home</th>
      <th>Number of people in household</th>
      <th>Number of people in household of 18+</th>
      <th>Number of people in household aged 13-17</th>
      <th>Number of people in household aged 5-12</th>
      <th>Number of people in household under age of 5</th>
      <th>Number of people in household aged 16-17</th>
      <th>Number of people in household aged 11-15</th>
      <th>Number of people in household aged 5-10</th>
      <th>Number of people in household aged 1-4</th>
      <th>Number of people in household under age of 1</th>
      <th>Living in household: partner, husband or wife</th>
      <th>Living in household: children</th>
      <th>Living in household: number of children</th>
      <th>Living in household: parents</th>
      <th>Living in household: number of parents</th>
      <th>Living in household: grandparents</th>
      <th>Living in household: number of grandparents</th>
      <th>Living in household: relatives</th>
      <th>Living in household: number of relatives</th>
      <th>Living in household: non relatives</th>
      <th>Living in household: number of non relatives</th>
      <th>What age did you complete your education</th>
      <th>What age did you complete your education (recoded in intervals)</th>
      <th>Had formal education</th>
      <th>Respondent - literate</th>
      <th>Highest educational level attained</th>
      <th>Educational level respondent: ISCED- code one digit</th>
      <th>Educational level respondent: ISCED-code two digits</th>
      <th>Educational level respondent: ISCED-code three digits</th>
      <th>Education (country specific)</th>
      <th>X025CSWVS</th>
      <th>Was the respondent literate</th>
      <th>Education level (recoded)</th>
      <th>Do you live with your parents</th>
      <th>House or apartment</th>
      <th>Employment status</th>
      <th>Employment/self-employment: last job</th>
      <th>Chief wage earner employed now</th>
      <th>Do you own your home or rent it</th>
      <th>Are you supervising someone</th>
      <th>Number of supervised people</th>
      <th>Number of supervised people (recoded)</th>
      <th>Number of supervised people, 3 cat</th>
      <th>Number of others working in the organization</th>
      <th>Number of others working in the organization (recoded)</th>
      <th>Number of employees</th>
      <th>Number of employees (recoded)</th>
      <th>Number of employees, 4 cat</th>
      <th>Job profession/industry (2 digit isco88)</th>
      <th>Job profession/industry (3 digit isco88)</th>
      <th>Job profession/industry (4 digit isco88)</th>
      <th>Profession/job</th>
      <th>Occupational status respondent - SIOPS</th>
      <th>Occupational status respondent - ISEI</th>
      <th>Occupational status respondent - egp11</th>
      <th>Occupational status respondent - European ESeC</th>
      <th>How long unemployed</th>
      <th>Respondent experienced unemployment longer than 3 months</th>
      <th>Dependency on social security during last 5 years respondent</th>
      <th>How many people work in your department-organization</th>
      <th>Do you or your spouse belong to a labour union</th>
      <th>Are you the chief wage earner in your house</th>
      <th>Is the chief wage earner employed now</th>
      <th>Profession/industry (2 digit isco88)</th>
      <th>Profession/industry (3 digit isco88)</th>
      <th>Profession/industry (4 digit isco88)</th>
      <th>Chief wage earner profession/job</th>
      <th>Family savings during past year</th>
      <th>Social class (subjective)</th>
      <th>Social class (subjective) with 6 categories</th>
      <th>Socio-economic status of respondent</th>
      <th>Scale of incomes</th>
      <th>Weekly household income</th>
      <th>Country specific: Weekly household income</th>
      <th>Monthly household income</th>
      <th>Country specific:Monthly household income</th>
      <th>Annual household income</th>
      <th>Country specific: Annual household income</th>
      <th>Income (country specific)</th>
      <th>Monthly household income (x1000), corrected for ppp in euros</th>
      <th>Income level</th>
      <th>Region where the interview was conducted</th>
      <th>Region: NUTS-1 code</th>
      <th>Region: NUTS-2 code</th>
      <th>Region: NUTS-3 code</th>
      <th>Region at age 14: country</th>
      <th>Region at age 14: NUTS-1 code</th>
      <th>Region at age 14: NUTS-2 code</th>
      <th>Region at age 14: NUTS-3 code</th>
      <th>X048WVS</th>
      <th>Size of town</th>
      <th>Size of town (country specific)</th>
      <th>Type of habitat</th>
      <th>Ethnic group</th>
      <th>Institution of occupation</th>
      <th>Nature of tasks: manual vs. Cognitive</th>
      <th>Nature of tasks: routine vs. Creative</th>
      <th>Nature of tasks: independence</th>
      <th>Post-Materialist index 12-item</th>
      <th>Post-Materialist index 4-item</th>
      <th>Autonomy Index</th>
      <th>Y010</th>
      <th>Y011</th>
      <th>Y012</th>
      <th>Y013</th>
      <th>Y014</th>
      <th>Y020</th>
      <th>Y021</th>
      <th>Y022</th>
      <th>Y023</th>
      <th>Y024</th>
      <th>TRADITIONAL/SECULAR RATIONAL VALUES</th>
      <th>SURVIVAL/SELF-EXPRESSION VALUES</th>
      <th>country_code</th>
      <th>country_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>1</td>
      <td>-4</td>
      <td>392</td>
      <td>392</td>
      <td>-4.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>3.920120e+09</td>
      <td>-4</td>
      <td>JP</td>
      <td>JP</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1.089722</td>
      <td>1.089722</td>
      <td>0.905084</td>
      <td>0.905084</td>
      <td>1.357627</td>
      <td>1.357627</td>
      <td>1981</td>
      <td>3.920121e+10</td>
      <td>3.920121e+10</td>
      <td>-4</td>
      <td>-4</td>
      <td>3921</td>
      <td>3921</td>
      <td>3921981</td>
      <td>3921981</td>
      <td>20150418</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>-2</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-2</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>1</td>
      <td>0</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>...</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>1933</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>49</td>
      <td>4</td>
      <td>2</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>17.0</td>
      <td>6.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-2</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>25</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>7</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>392004</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>392001</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>NaN</td>
      <td>2</td>
      <td>-1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.901</td>
      <td>NaN</td>
      <td>0.553333</td>
      <td>NaN</td>
      <td>0.666667</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.35248</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>392.0</td>
      <td>Japan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>-4</td>
      <td>392</td>
      <td>392</td>
      <td>-4.0</td>
      <td>2.0</td>
      <td>2</td>
      <td>3.920120e+09</td>
      <td>-4</td>
      <td>JP</td>
      <td>JP</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1.089722</td>
      <td>1.089722</td>
      <td>0.905084</td>
      <td>0.905084</td>
      <td>1.357627</td>
      <td>1.357627</td>
      <td>1981</td>
      <td>3.920121e+10</td>
      <td>3.920121e+10</td>
      <td>-4</td>
      <td>-4</td>
      <td>3921</td>
      <td>3921</td>
      <td>3921981</td>
      <td>3921981</td>
      <td>20150418</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>1</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>1</td>
      <td>0</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>0</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>...</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>1933</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>49</td>
      <td>4</td>
      <td>2</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>14.0</td>
      <td>3.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-2</td>
      <td>-4</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>41</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>25</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-2</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>392004</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>392001</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>NaN</td>
      <td>1</td>
      <td>-1.0</td>
      <td>0.36594</td>
      <td>0.68776</td>
      <td>0.556</td>
      <td>0.0</td>
      <td>0.220000</td>
      <td>0.398303</td>
      <td>0.666667</td>
      <td>NaN</td>
      <td>0.259259</td>
      <td>0.13600</td>
      <td>0.515189</td>
      <td>-0.548588</td>
      <td>392.0</td>
      <td>Japan</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>-4</td>
      <td>392</td>
      <td>392</td>
      <td>-4.0</td>
      <td>3.0</td>
      <td>3</td>
      <td>3.920120e+09</td>
      <td>-4</td>
      <td>JP</td>
      <td>JP</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1.089722</td>
      <td>1.089722</td>
      <td>0.905084</td>
      <td>0.905084</td>
      <td>1.357627</td>
      <td>1.357627</td>
      <td>1981</td>
      <td>3.920121e+10</td>
      <td>3.920121e+10</td>
      <td>-4</td>
      <td>-4</td>
      <td>3921</td>
      <td>3921</td>
      <td>3921981</td>
      <td>3921981</td>
      <td>20150418</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>2</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>...</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>1901</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>81</td>
      <td>6</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>15.0</td>
      <td>4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-2</td>
      <td>-4</td>
      <td>7</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>41</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>41</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>392004</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>392001</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>NaN</td>
      <td>1</td>
      <td>2.0</td>
      <td>0.34294</td>
      <td>0.68776</td>
      <td>0.244</td>
      <td>0.0</td>
      <td>0.440000</td>
      <td>0.141896</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.13600</td>
      <td>0.241489</td>
      <td>-1.603452</td>
      <td>392.0</td>
      <td>Japan</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
      <td>-4</td>
      <td>392</td>
      <td>392</td>
      <td>-4.0</td>
      <td>4.0</td>
      <td>4</td>
      <td>3.920120e+09</td>
      <td>-4</td>
      <td>JP</td>
      <td>JP</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1.089722</td>
      <td>1.089722</td>
      <td>0.905084</td>
      <td>0.905084</td>
      <td>1.357627</td>
      <td>1.357627</td>
      <td>1981</td>
      <td>3.920121e+10</td>
      <td>3.920121e+10</td>
      <td>-4</td>
      <td>-4</td>
      <td>3921</td>
      <td>3921</td>
      <td>3921981</td>
      <td>3921981</td>
      <td>20150418</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>2</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>...</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>1901</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>81</td>
      <td>6</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>15.0</td>
      <td>4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-2</td>
      <td>-4</td>
      <td>7</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>41</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>41</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>392004</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>392001</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>NaN</td>
      <td>1</td>
      <td>2.0</td>
      <td>0.34294</td>
      <td>0.68776</td>
      <td>0.244</td>
      <td>0.0</td>
      <td>0.440000</td>
      <td>0.141896</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.13600</td>
      <td>0.241489</td>
      <td>-1.603452</td>
      <td>392.0</td>
      <td>Japan</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>1</td>
      <td>-4</td>
      <td>392</td>
      <td>392</td>
      <td>-4.0</td>
      <td>5.0</td>
      <td>5</td>
      <td>3.920120e+09</td>
      <td>-4</td>
      <td>JP</td>
      <td>JP</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1.089722</td>
      <td>1.089722</td>
      <td>0.905084</td>
      <td>0.905084</td>
      <td>1.357627</td>
      <td>1.357627</td>
      <td>1981</td>
      <td>3.920121e+10</td>
      <td>3.920121e+10</td>
      <td>-4</td>
      <td>-4</td>
      <td>3921</td>
      <td>3921</td>
      <td>3921981</td>
      <td>3921981</td>
      <td>20150418</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>2</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>1</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>0</td>
      <td>1</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>...</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>2</td>
      <td>1917</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>65</td>
      <td>6</td>
      <td>3</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>5</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>4</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>17.0</td>
      <td>6.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4.0</td>
      <td>-2</td>
      <td>-4</td>
      <td>5</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>41</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>41</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>8</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>392004</td>
      <td>-4</td>
      <td>-4</td>
      <td>1</td>
      <td>392001</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>NaN</td>
      <td>1</td>
      <td>-1.0</td>
      <td>0.43269</td>
      <td>0.68776</td>
      <td>0.823</td>
      <td>0.0</td>
      <td>0.220000</td>
      <td>0.298340</td>
      <td>0.333333</td>
      <td>NaN</td>
      <td>0.222222</td>
      <td>0.13600</td>
      <td>0.129725</td>
      <td>-0.377730</td>
      <td>392.0</td>
      <td>Japan</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1412 columns</p>
</div>




```python
df['country_year'] = df['country_name'].astype(str) + '_' + df['Year survey'].astype(str)
```

#### Now, zoom in on the columns of interest:  What is important in a job?


```python
job_qs = important_in_a_job
baseline_qs = ['Wave', 'Year survey', 'Country/region', 'country_name', 'country_year']
job_df = df[baseline_qs + job_qs]
```


```python
job_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Wave</th>
      <th>Year survey</th>
      <th>Country/region</th>
      <th>country_name</th>
      <th>country_year</th>
      <th>Important in a job: good pay</th>
      <th>Important in a job: not too much pressure</th>
      <th>Important in a job: good job security</th>
      <th>Important in a job: a respected job</th>
      <th>Important in a job: good hours</th>
      <th>Important in a job: an opportunity to use initiative</th>
      <th>Important in a job: generous holidays</th>
      <th>Important in a job: that you can achieve something</th>
      <th>Important in a job: a responsible job</th>
      <th>Important in a job: a job that is interesting</th>
      <th>Important in a job: a job that meets one´s abilities</th>
      <th>Important in a job: pleasant people to work with</th>
      <th>Important in a job: good chances for promotion</th>
      <th>Important in a job: a useful job for society</th>
      <th>Important in a job: meeting people</th>
      <th>Important in a job: good physical working conditions</th>
      <th>Important in a job: to have time off at the weekends</th>
      <th>Important in a job: learning new skills</th>
      <th>Important in a job: family friendly</th>
      <th>Important in a job: have a say</th>
      <th>Important in a job: people treated equally</th>
      <th>Important in a job: none of these</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1981</td>
      <td>392</td>
      <td>Japan</td>
      <td>Japan_1981</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1981</td>
      <td>392</td>
      <td>Japan</td>
      <td>Japan_1981</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1981</td>
      <td>392</td>
      <td>Japan</td>
      <td>Japan_1981</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1981</td>
      <td>392</td>
      <td>Japan</td>
      <td>Japan_1981</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1981</td>
      <td>392</td>
      <td>Japan</td>
      <td>Japan_1981</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
      <td>-4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Count nulls
job_df.isnull().sum().sum()
```




    3780




```python
# Count values greater than or equal to zero (from the data dictionary, negative numbers indicate missing data)
(job_df[job_qs] >= 0).sum()
```




    Important in a job: good pay                            151669
    Important in a job: not too much pressure               151669
    Important in a job: good job security                   151669
    Important in a job: a respected job                     151669
    Important in a job: good hours                          148933
    Important in a job: an opportunity to use initiative    149647
    Important in a job: generous holidays                   150699
    Important in a job: that you can achieve something      148933
    Important in a job: a responsible job                   151669
    Important in a job: a job that is interesting           148933
    Important in a job: a job that meets one´s abilities    150516
    Important in a job: pleasant people to work with         20175
    Important in a job: good chances for promotion           20175
    Important in a job: a useful job for society             20175
    Important in a job: meeting people                       17439
    Important in a job: good physical working conditions         0
    Important in a job: to have time off at the weekends         0
    Important in a job: learning new skills                      0
    Important in a job: family friendly                          0
    Important in a job: have a say                               0
    Important in a job: people treated equally                   0
    Important in a job: none of these                        19174
    dtype: int64




```python
# Focusing in on the columns with the fewest missing data points:
refined_job_qs = ['Important in a job: good pay',
 'Important in a job: not too much pressure',
 'Important in a job: good job security',
 'Important in a job: a respected job',
 'Important in a job: good hours',
 'Important in a job: an opportunity to use initiative',
 'Important in a job: generous holidays',
 'Important in a job: that you can achieve something',
 'Important in a job: a responsible job',
 'Important in a job: a job that is interesting',
 'Important in a job: a job that meets one´s abilities']

refined_job_df = df[baseline_qs + refined_job_qs].copy()

(refined_job_df[refined_job_qs] >= 0).sum()
```




    Important in a job: good pay                            151669
    Important in a job: not too much pressure               151669
    Important in a job: good job security                   151669
    Important in a job: a respected job                     151669
    Important in a job: good hours                          148933
    Important in a job: an opportunity to use initiative    149647
    Important in a job: generous holidays                   150699
    Important in a job: that you can achieve something      148933
    Important in a job: a responsible job                   151669
    Important in a job: a job that is interesting           148933
    Important in a job: a job that meets one´s abilities    150516
    dtype: int64




```python
# Check how many people partially answered the questions
refined_job_df['number_of_jobs_qs_answered'] = (refined_job_df[refined_job_qs] >= 0).sum(axis = 1)
```


```python
refined_job_df['number_of_jobs_qs_answered'].value_counts()
```




    0     189602
    11    144788
    10      4145
    8       2736
    Name: number_of_jobs_qs_answered, dtype: int64




```python
# Overall, most people answered all the questions (144k) or none (189k). 
# A small proportion (~6k) responsed to some but not all of the questions
# Dropping respondants who did not answer all questiosn will improve consistency and reduce bias, for a relatively small data loss
# Consequently, we will focus only on those respondants who answered all questions
mask = refined_job_df['number_of_jobs_qs_answered'] == 11
refined_job_df = refined_job_df[mask]
```

## 3. EDA

### Preview the data and look for correlations (clustering of input variables / multicolinearity)


```python
# View dataframe
refined_job_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Wave</th>
      <th>Year survey</th>
      <th>Country/region</th>
      <th>country_name</th>
      <th>country_year</th>
      <th>Important in a job: good pay</th>
      <th>Important in a job: not too much pressure</th>
      <th>Important in a job: good job security</th>
      <th>Important in a job: a respected job</th>
      <th>Important in a job: good hours</th>
      <th>Important in a job: an opportunity to use initiative</th>
      <th>Important in a job: generous holidays</th>
      <th>Important in a job: that you can achieve something</th>
      <th>Important in a job: a responsible job</th>
      <th>Important in a job: a job that is interesting</th>
      <th>Important in a job: a job that meets one´s abilities</th>
      <th>number_of_jobs_qs_answered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1981</td>
      <td>392</td>
      <td>Japan</td>
      <td>Japan_1981</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1981</td>
      <td>392</td>
      <td>Japan</td>
      <td>Japan_1981</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1981</td>
      <td>392</td>
      <td>Japan</td>
      <td>Japan_1981</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1981</td>
      <td>392</td>
      <td>Japan</td>
      <td>Japan_1981</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1981</td>
      <td>392</td>
      <td>Japan</td>
      <td>Japan_1981</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Look at correlation matrix between different questions
sns.heatmap(refined_job_df[refined_job_qs].corr(), annot = True);
```


![png](/images/World_Values--Full_dataset_files/World_Values--Full_dataset_32_0.png)


#### Though correlations are all < 0.5, we can see correlated clusters of variables:
- Achieving something, responsible job and opportunity to use initiative (correlations of ~0.4)
- Good hours, generous holidays, not too much pressure (correlations of ~0.35)

### Understand sample size


```python
# View dataframe
refined_job_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Wave</th>
      <th>Year survey</th>
      <th>Country/region</th>
      <th>country_name</th>
      <th>country_year</th>
      <th>Important in a job: good pay</th>
      <th>Important in a job: not too much pressure</th>
      <th>Important in a job: good job security</th>
      <th>Important in a job: a respected job</th>
      <th>Important in a job: good hours</th>
      <th>Important in a job: an opportunity to use initiative</th>
      <th>Important in a job: generous holidays</th>
      <th>Important in a job: that you can achieve something</th>
      <th>Important in a job: a responsible job</th>
      <th>Important in a job: a job that is interesting</th>
      <th>Important in a job: a job that meets one´s abilities</th>
      <th>number_of_jobs_qs_answered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1981</td>
      <td>392</td>
      <td>Japan</td>
      <td>Japan_1981</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1981</td>
      <td>392</td>
      <td>Japan</td>
      <td>Japan_1981</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1981</td>
      <td>392</td>
      <td>Japan</td>
      <td>Japan_1981</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1981</td>
      <td>392</td>
      <td>Japan</td>
      <td>Japan_1981</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1981</td>
      <td>392</td>
      <td>Japan</td>
      <td>Japan_1981</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Summary statistics
country_counts = refined_job_df.groupby('country_year').count()
country_counts[refined_job_qs].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Important in a job: good pay</th>
      <th>Important in a job: not too much pressure</th>
      <th>Important in a job: good job security</th>
      <th>Important in a job: a respected job</th>
      <th>Important in a job: good hours</th>
      <th>Important in a job: an opportunity to use initiative</th>
      <th>Important in a job: generous holidays</th>
      <th>Important in a job: that you can achieve something</th>
      <th>Important in a job: a responsible job</th>
      <th>Important in a job: a job that is interesting</th>
      <th>Important in a job: a job that meets one´s abilities</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>104.000000</td>
      <td>104.000000</td>
      <td>104.000000</td>
      <td>104.000000</td>
      <td>104.000000</td>
      <td>104.000000</td>
      <td>104.000000</td>
      <td>104.000000</td>
      <td>104.000000</td>
      <td>104.000000</td>
      <td>104.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1392.192308</td>
      <td>1392.192308</td>
      <td>1392.192308</td>
      <td>1392.192308</td>
      <td>1392.192308</td>
      <td>1392.192308</td>
      <td>1392.192308</td>
      <td>1392.192308</td>
      <td>1392.192308</td>
      <td>1392.192308</td>
      <td>1392.192308</td>
    </tr>
    <tr>
      <th>std</th>
      <td>556.828137</td>
      <td>556.828137</td>
      <td>556.828137</td>
      <td>556.828137</td>
      <td>556.828137</td>
      <td>556.828137</td>
      <td>556.828137</td>
      <td>556.828137</td>
      <td>556.828137</td>
      <td>556.828137</td>
      <td>556.828137</td>
    </tr>
    <tr>
      <th>min</th>
      <td>417.000000</td>
      <td>417.000000</td>
      <td>417.000000</td>
      <td>417.000000</td>
      <td>417.000000</td>
      <td>417.000000</td>
      <td>417.000000</td>
      <td>417.000000</td>
      <td>417.000000</td>
      <td>417.000000</td>
      <td>417.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1007.750000</td>
      <td>1007.750000</td>
      <td>1007.750000</td>
      <td>1007.750000</td>
      <td>1007.750000</td>
      <td>1007.750000</td>
      <td>1007.750000</td>
      <td>1007.750000</td>
      <td>1007.750000</td>
      <td>1007.750000</td>
      <td>1007.750000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1200.500000</td>
      <td>1200.500000</td>
      <td>1200.500000</td>
      <td>1200.500000</td>
      <td>1200.500000</td>
      <td>1200.500000</td>
      <td>1200.500000</td>
      <td>1200.500000</td>
      <td>1200.500000</td>
      <td>1200.500000</td>
      <td>1200.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1532.000000</td>
      <td>1532.000000</td>
      <td>1532.000000</td>
      <td>1532.000000</td>
      <td>1532.000000</td>
      <td>1532.000000</td>
      <td>1532.000000</td>
      <td>1532.000000</td>
      <td>1532.000000</td>
      <td>1532.000000</td>
      <td>1532.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3401.000000</td>
      <td>3401.000000</td>
      <td>3401.000000</td>
      <td>3401.000000</td>
      <td>3401.000000</td>
      <td>3401.000000</td>
      <td>3401.000000</td>
      <td>3401.000000</td>
      <td>3401.000000</td>
      <td>3401.000000</td>
      <td>3401.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Summary
print(f'Each question was, on average, answered by {country_counts.iloc[:,0].mean():.0f} people in each wave, with a \
min of {country_counts.iloc[:,0].min():.0f} and a max of {country_counts.iloc[:,0].max():.0f}')
```

    Each question was, on average, answered by 1392 people in each wave, with a min of 417 and a max of 3401


### See how responses evolve on a country-by-country level over time


```python
# Aggregate
country_averages = refined_job_df.groupby('country_year').mean()
country_averages['country_year'] = country_averages.index
country_averages.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Wave</th>
      <th>Year survey</th>
      <th>Country/region</th>
      <th>Important in a job: good pay</th>
      <th>Important in a job: not too much pressure</th>
      <th>Important in a job: good job security</th>
      <th>Important in a job: a respected job</th>
      <th>Important in a job: good hours</th>
      <th>Important in a job: an opportunity to use initiative</th>
      <th>Important in a job: generous holidays</th>
      <th>Important in a job: that you can achieve something</th>
      <th>Important in a job: a responsible job</th>
      <th>Important in a job: a job that is interesting</th>
      <th>Important in a job: a job that meets one´s abilities</th>
      <th>number_of_jobs_qs_answered</th>
      <th>country_year</th>
    </tr>
    <tr>
      <th>country_year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Albania_1998</th>
      <td>3.0</td>
      <td>1998</td>
      <td>8.0</td>
      <td>0.994995</td>
      <td>0.372372</td>
      <td>0.853854</td>
      <td>0.417417</td>
      <td>0.610611</td>
      <td>0.426426</td>
      <td>0.533534</td>
      <td>0.521522</td>
      <td>0.173173</td>
      <td>0.458458</td>
      <td>0.888889</td>
      <td>11.0</td>
      <td>Albania_1998</td>
    </tr>
    <tr>
      <th>Albania_2002</th>
      <td>4.0</td>
      <td>2002</td>
      <td>8.0</td>
      <td>0.952000</td>
      <td>0.455000</td>
      <td>0.812000</td>
      <td>0.645000</td>
      <td>0.550000</td>
      <td>0.397000</td>
      <td>0.483000</td>
      <td>0.513000</td>
      <td>0.254000</td>
      <td>0.411000</td>
      <td>0.294000</td>
      <td>11.0</td>
      <td>Albania_2002</td>
    </tr>
    <tr>
      <th>Algeria_2002</th>
      <td>4.0</td>
      <td>2002</td>
      <td>12.0</td>
      <td>0.897816</td>
      <td>0.595164</td>
      <td>0.863495</td>
      <td>0.719969</td>
      <td>0.511700</td>
      <td>0.435257</td>
      <td>0.205148</td>
      <td>0.602964</td>
      <td>0.464119</td>
      <td>0.565523</td>
      <td>0.701248</td>
      <td>11.0</td>
      <td>Algeria_2002</td>
    </tr>
    <tr>
      <th>Argentina_1984</th>
      <td>1.0</td>
      <td>1984</td>
      <td>32.0</td>
      <td>0.862687</td>
      <td>0.366169</td>
      <td>0.528358</td>
      <td>0.264677</td>
      <td>0.560199</td>
      <td>0.468657</td>
      <td>0.371144</td>
      <td>0.338308</td>
      <td>0.331343</td>
      <td>0.383085</td>
      <td>0.352239</td>
      <td>11.0</td>
      <td>Argentina_1984</td>
    </tr>
    <tr>
      <th>Argentina_1991</th>
      <td>2.0</td>
      <td>1991</td>
      <td>32.0</td>
      <td>0.840319</td>
      <td>0.407186</td>
      <td>0.632735</td>
      <td>0.428144</td>
      <td>0.435130</td>
      <td>0.546906</td>
      <td>0.299401</td>
      <td>0.501996</td>
      <td>0.581836</td>
      <td>0.506986</td>
      <td>0.584830</td>
      <td>11.0</td>
      <td>Argentina_1991</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Add country name column
country_averages['country_name'] = [country_averages.index[i][0:country_averages.index[i].find('_')] for i in range(0, len(country_averages.index))]
```


```python
# Assess data availability
country_averages['country_name'].value_counts()[0:10]
```




    Mexico       5
    Argentina    4
    Japan        4
    Turkey       3
    Chile        3
    Spain        3
    China        3
    India        3
    Moldova      2
    Hungary      2
    Name: country_name, dtype: int64




```python
country_averages['country_name'].value_counts().count()
```




    64



#### Of the 64 countries included in the datset, 8 have data from three or more waves; we will focus on these


```python
# Create sub-dfs for each of these countries
longitudinal_countries = ['Mexico', 'Japan', 'Argentina', 'India', 'China', 'Chile', 'Turkey','Spain']
```


```python
for index, df in country_averages.groupby(country_averages.index):
    if index == 'Mexico':
        display(df)
```


```python
# Plot evolution of values over time

country_averages.index = country_averages['country_name']

n_cols = 2
n_rows = int(len(longitudinal_countries)//n_cols)
figure, ax = plt.subplots(nrows = n_rows, ncols=n_cols, figsize=(18,36))
i = 0 # this is a counter to tell matplotlib which axis to plot on
_cmap = plt.get_cmap('tab20')

for index, dataframe in country_averages.groupby(country_averages.index):
    if index in longitudinal_countries:
        ax[i//n_cols][i%n_cols].set_title(f"\n {index} \n", fontsize = 15)
        for j in range(0, len(refined_job_qs)): # this for loop is used to improve line colors on charts
            ax[i//n_cols][i%n_cols].plot(dataframe['Year survey'], dataframe[refined_job_qs[j]], c=_cmap.colors[j])
        ax[i//n_cols][i%n_cols].set_xlabel('Year', horizontalalignment = 'right')
        ax[i//n_cols][i%n_cols].set_ylabel('Importance score')
        if i == 0:
            ax[i//n_cols][i%n_cols].legend(refined_job_qs, )
        i +=1
```


![png](/images/World_Values--Full_dataset_files/World_Values--Full_dataset_46_0.png)


#### Normalizing our data

In some countries (e.g., Japan, Mexico) the variables tend to trend up and down together. This could be an artefact of how the survey was administered: the value for each cell is the percent of respondants who mentioned something as important. In some years, the interviews may have been longer or more comprehensive than others. To account for this fact, we can look at the number of times an attribute was mentioned as a % of the total.


```python
country_averages['total_mentions'] = country_averages[refined_job_qs].sum(axis = 1)
country_averages.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Wave</th>
      <th>Year survey</th>
      <th>Country/region</th>
      <th>Important in a job: good pay</th>
      <th>Important in a job: not too much pressure</th>
      <th>Important in a job: good job security</th>
      <th>Important in a job: a respected job</th>
      <th>Important in a job: good hours</th>
      <th>Important in a job: an opportunity to use initiative</th>
      <th>Important in a job: generous holidays</th>
      <th>Important in a job: that you can achieve something</th>
      <th>Important in a job: a responsible job</th>
      <th>Important in a job: a job that is interesting</th>
      <th>Important in a job: a job that meets one´s abilities</th>
      <th>number_of_jobs_qs_answered</th>
      <th>country_year</th>
      <th>country_name</th>
      <th>total_mentions</th>
    </tr>
    <tr>
      <th>country_name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Albania</th>
      <td>3.0</td>
      <td>1998</td>
      <td>8.0</td>
      <td>0.994995</td>
      <td>0.372372</td>
      <td>0.853854</td>
      <td>0.417417</td>
      <td>0.610611</td>
      <td>0.426426</td>
      <td>0.533534</td>
      <td>0.521522</td>
      <td>0.173173</td>
      <td>0.458458</td>
      <td>0.888889</td>
      <td>11.0</td>
      <td>Albania_1998</td>
      <td>Albania</td>
      <td>6.251251</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>4.0</td>
      <td>2002</td>
      <td>8.0</td>
      <td>0.952000</td>
      <td>0.455000</td>
      <td>0.812000</td>
      <td>0.645000</td>
      <td>0.550000</td>
      <td>0.397000</td>
      <td>0.483000</td>
      <td>0.513000</td>
      <td>0.254000</td>
      <td>0.411000</td>
      <td>0.294000</td>
      <td>11.0</td>
      <td>Albania_2002</td>
      <td>Albania</td>
      <td>5.766000</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>4.0</td>
      <td>2002</td>
      <td>12.0</td>
      <td>0.897816</td>
      <td>0.595164</td>
      <td>0.863495</td>
      <td>0.719969</td>
      <td>0.511700</td>
      <td>0.435257</td>
      <td>0.205148</td>
      <td>0.602964</td>
      <td>0.464119</td>
      <td>0.565523</td>
      <td>0.701248</td>
      <td>11.0</td>
      <td>Algeria_2002</td>
      <td>Algeria</td>
      <td>6.562402</td>
    </tr>
    <tr>
      <th>Argentina</th>
      <td>1.0</td>
      <td>1984</td>
      <td>32.0</td>
      <td>0.862687</td>
      <td>0.366169</td>
      <td>0.528358</td>
      <td>0.264677</td>
      <td>0.560199</td>
      <td>0.468657</td>
      <td>0.371144</td>
      <td>0.338308</td>
      <td>0.331343</td>
      <td>0.383085</td>
      <td>0.352239</td>
      <td>11.0</td>
      <td>Argentina_1984</td>
      <td>Argentina</td>
      <td>4.826866</td>
    </tr>
    <tr>
      <th>Argentina</th>
      <td>2.0</td>
      <td>1991</td>
      <td>32.0</td>
      <td>0.840319</td>
      <td>0.407186</td>
      <td>0.632735</td>
      <td>0.428144</td>
      <td>0.435130</td>
      <td>0.546906</td>
      <td>0.299401</td>
      <td>0.501996</td>
      <td>0.581836</td>
      <td>0.506986</td>
      <td>0.584830</td>
      <td>11.0</td>
      <td>Argentina_1991</td>
      <td>Argentina</td>
      <td>5.765469</td>
    </tr>
  </tbody>
</table>
</div>




```python
for column in refined_job_qs:
    country_averages[f'normalized {column}']=country_averages[column]/country_averages['total_mentions'] * 100 
    # * 100 to get pct
```


```python
normalized_refined_job_qs = [f'normalized {column}'for column in refined_job_qs]
country_averages['normalized total_mentions'] = country_averages[normalized_refined_job_qs].sum(axis = 1)
```


```python
country_averages.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Wave</th>
      <th>Year survey</th>
      <th>Country/region</th>
      <th>Important in a job: good pay</th>
      <th>Important in a job: not too much pressure</th>
      <th>Important in a job: good job security</th>
      <th>Important in a job: a respected job</th>
      <th>Important in a job: good hours</th>
      <th>Important in a job: an opportunity to use initiative</th>
      <th>Important in a job: generous holidays</th>
      <th>Important in a job: that you can achieve something</th>
      <th>Important in a job: a responsible job</th>
      <th>Important in a job: a job that is interesting</th>
      <th>Important in a job: a job that meets one´s abilities</th>
      <th>number_of_jobs_qs_answered</th>
      <th>country_year</th>
      <th>country_name</th>
      <th>total_mentions</th>
      <th>normalized Important in a job: good pay</th>
      <th>normalized Important in a job: not too much pressure</th>
      <th>normalized Important in a job: good job security</th>
      <th>normalized Important in a job: a respected job</th>
      <th>normalized Important in a job: good hours</th>
      <th>normalized Important in a job: an opportunity to use initiative</th>
      <th>normalized Important in a job: generous holidays</th>
      <th>normalized Important in a job: that you can achieve something</th>
      <th>normalized Important in a job: a responsible job</th>
      <th>normalized Important in a job: a job that is interesting</th>
      <th>normalized Important in a job: a job that meets one´s abilities</th>
      <th>normalized total_mentions</th>
    </tr>
    <tr>
      <th>country_name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Albania</th>
      <td>3.0</td>
      <td>1998</td>
      <td>8.0</td>
      <td>0.994995</td>
      <td>0.372372</td>
      <td>0.853854</td>
      <td>0.417417</td>
      <td>0.610611</td>
      <td>0.426426</td>
      <td>0.533534</td>
      <td>0.521522</td>
      <td>0.173173</td>
      <td>0.458458</td>
      <td>0.888889</td>
      <td>11.0</td>
      <td>Albania_1998</td>
      <td>Albania</td>
      <td>6.251251</td>
      <td>15.916733</td>
      <td>5.956765</td>
      <td>13.658927</td>
      <td>6.677342</td>
      <td>9.767814</td>
      <td>6.821457</td>
      <td>8.534828</td>
      <td>8.342674</td>
      <td>2.770216</td>
      <td>7.333867</td>
      <td>14.219376</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>4.0</td>
      <td>2002</td>
      <td>8.0</td>
      <td>0.952000</td>
      <td>0.455000</td>
      <td>0.812000</td>
      <td>0.645000</td>
      <td>0.550000</td>
      <td>0.397000</td>
      <td>0.483000</td>
      <td>0.513000</td>
      <td>0.254000</td>
      <td>0.411000</td>
      <td>0.294000</td>
      <td>11.0</td>
      <td>Albania_2002</td>
      <td>Albania</td>
      <td>5.766000</td>
      <td>16.510579</td>
      <td>7.891086</td>
      <td>14.082553</td>
      <td>11.186264</td>
      <td>9.538675</td>
      <td>6.885189</td>
      <td>8.376691</td>
      <td>8.896982</td>
      <td>4.405134</td>
      <td>7.127992</td>
      <td>5.098855</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>4.0</td>
      <td>2002</td>
      <td>12.0</td>
      <td>0.897816</td>
      <td>0.595164</td>
      <td>0.863495</td>
      <td>0.719969</td>
      <td>0.511700</td>
      <td>0.435257</td>
      <td>0.205148</td>
      <td>0.602964</td>
      <td>0.464119</td>
      <td>0.565523</td>
      <td>0.701248</td>
      <td>11.0</td>
      <td>Algeria_2002</td>
      <td>Algeria</td>
      <td>6.562402</td>
      <td>13.681208</td>
      <td>9.069298</td>
      <td>13.158208</td>
      <td>10.971116</td>
      <td>7.797456</td>
      <td>6.632592</td>
      <td>3.126114</td>
      <td>9.188161</td>
      <td>7.072388</td>
      <td>8.617616</td>
      <td>10.685843</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Argentina</th>
      <td>1.0</td>
      <td>1984</td>
      <td>32.0</td>
      <td>0.862687</td>
      <td>0.366169</td>
      <td>0.528358</td>
      <td>0.264677</td>
      <td>0.560199</td>
      <td>0.468657</td>
      <td>0.371144</td>
      <td>0.338308</td>
      <td>0.331343</td>
      <td>0.383085</td>
      <td>0.352239</td>
      <td>11.0</td>
      <td>Argentina_1984</td>
      <td>Argentina</td>
      <td>4.826866</td>
      <td>17.872604</td>
      <td>7.586065</td>
      <td>10.946197</td>
      <td>5.483405</td>
      <td>11.605854</td>
      <td>9.709338</td>
      <td>7.689136</td>
      <td>7.008864</td>
      <td>6.864564</td>
      <td>7.936508</td>
      <td>7.297464</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Argentina</th>
      <td>2.0</td>
      <td>1991</td>
      <td>32.0</td>
      <td>0.840319</td>
      <td>0.407186</td>
      <td>0.632735</td>
      <td>0.428144</td>
      <td>0.435130</td>
      <td>0.546906</td>
      <td>0.299401</td>
      <td>0.501996</td>
      <td>0.581836</td>
      <td>0.506986</td>
      <td>0.584830</td>
      <td>11.0</td>
      <td>Argentina_1991</td>
      <td>Argentina</td>
      <td>5.765469</td>
      <td>14.575039</td>
      <td>7.062489</td>
      <td>10.974554</td>
      <td>7.426000</td>
      <td>7.547170</td>
      <td>9.485892</td>
      <td>5.193007</td>
      <td>8.706941</td>
      <td>10.091743</td>
      <td>8.793491</td>
      <td>10.143673</td>
      <td>100.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot evolution of values over time

country_averages.index = country_averages['country_name']

n_cols = 2
n_rows = int(len(longitudinal_countries)//n_cols)
figure, ax = plt.subplots(nrows = n_rows, ncols=n_cols, figsize=(18,30))
i = 0 # this is a counter to tell matplotlib which axis to plot on
_cmap = plt.get_cmap('tab20')

for index, dataframe in country_averages.groupby(country_averages.index):
    if index in longitudinal_countries:
        ax[i//n_cols][i%n_cols].set_title(f"\n {index} \n", fontsize = 15)
        for j in range(0, len(normalized_refined_job_qs)): # this for loop is used to improve line colors on charts
            ax[i//n_cols][i%n_cols].plot(dataframe['Year survey'], dataframe[normalized_refined_job_qs[j]], c=_cmap.colors[j])
        ax[i//n_cols][i%n_cols].set_xlabel('Year', horizontalalignment = 'right')
        # ax[i//n_cols][i%n_cols].set_xlim(xmin = 1980)
        ax[i//n_cols][i%n_cols].set_ylabel('Percent of total mentions')
        ax[i//n_cols][i%n_cols].legend(refined_job_qs, loc = 'lower left')
        i +=1
```


![png](/images/World_Values--Full_dataset_files/World_Values--Full_dataset_52_0.png)


#### Feature engineering for clustering analysis

As noted above, we can see that certain variables are correlated:
- Achieving something, responsible job and opportunity to use initiative (correlations of ~0.4)
- Good hours, generous holidays, not too much pressure (correlations of ~0.35)

Using a methodology similar to PCA, we can create new features to reduce the dimensionality of our features.


```python
# Look at correlation matrix between different questions
sns.heatmap(country_averages[normalized_refined_job_qs].corr(), annot = True);
```


![png](/images/World_Values--Full_dataset_files/World_Values--Full_dataset_54_0.png)



```python
# Create new features

# Achieving_responsible_initiative

country_averages['normalized achieving_responsible_initiative'] = \
    country_averages['normalized Important in a job: that you can achieve something'] + \
    country_averages['normalized Important in a job: a responsible job'] + \
    country_averages['normalized Important in a job: an opportunity to use initiative']

# Hours_holidays_pressure

country_averages['normalized hours_holidays_pressure'] = \
    country_averages['normalized Important in a job: good hours'] + \
    country_averages['normalized Important in a job: generous holidays'] + \
    country_averages['normalized Important in a job: not too much pressure']
```


```python
country_averages.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Wave</th>
      <th>Year survey</th>
      <th>Country/region</th>
      <th>Important in a job: good pay</th>
      <th>Important in a job: not too much pressure</th>
      <th>Important in a job: good job security</th>
      <th>Important in a job: a respected job</th>
      <th>Important in a job: good hours</th>
      <th>Important in a job: an opportunity to use initiative</th>
      <th>Important in a job: generous holidays</th>
      <th>Important in a job: that you can achieve something</th>
      <th>Important in a job: a responsible job</th>
      <th>Important in a job: a job that is interesting</th>
      <th>Important in a job: a job that meets one´s abilities</th>
      <th>number_of_jobs_qs_answered</th>
      <th>country_year</th>
      <th>country_name</th>
      <th>total_mentions</th>
      <th>normalized Important in a job: good pay</th>
      <th>normalized Important in a job: not too much pressure</th>
      <th>normalized Important in a job: good job security</th>
      <th>normalized Important in a job: a respected job</th>
      <th>normalized Important in a job: good hours</th>
      <th>normalized Important in a job: an opportunity to use initiative</th>
      <th>normalized Important in a job: generous holidays</th>
      <th>normalized Important in a job: that you can achieve something</th>
      <th>normalized Important in a job: a responsible job</th>
      <th>normalized Important in a job: a job that is interesting</th>
      <th>normalized Important in a job: a job that meets one´s abilities</th>
      <th>normalized total_mentions</th>
      <th>normalized achieving_responsible_initiative</th>
      <th>normalized hours_holidays_pressure</th>
    </tr>
    <tr>
      <th>country_name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Albania</th>
      <td>3.0</td>
      <td>1998</td>
      <td>8.0</td>
      <td>0.994995</td>
      <td>0.372372</td>
      <td>0.853854</td>
      <td>0.417417</td>
      <td>0.610611</td>
      <td>0.426426</td>
      <td>0.533534</td>
      <td>0.521522</td>
      <td>0.173173</td>
      <td>0.458458</td>
      <td>0.888889</td>
      <td>11.0</td>
      <td>Albania_1998</td>
      <td>Albania</td>
      <td>6.251251</td>
      <td>15.916733</td>
      <td>5.956765</td>
      <td>13.658927</td>
      <td>6.677342</td>
      <td>9.767814</td>
      <td>6.821457</td>
      <td>8.534828</td>
      <td>8.342674</td>
      <td>2.770216</td>
      <td>7.333867</td>
      <td>14.219376</td>
      <td>100.0</td>
      <td>17.934347</td>
      <td>24.259408</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>4.0</td>
      <td>2002</td>
      <td>8.0</td>
      <td>0.952000</td>
      <td>0.455000</td>
      <td>0.812000</td>
      <td>0.645000</td>
      <td>0.550000</td>
      <td>0.397000</td>
      <td>0.483000</td>
      <td>0.513000</td>
      <td>0.254000</td>
      <td>0.411000</td>
      <td>0.294000</td>
      <td>11.0</td>
      <td>Albania_2002</td>
      <td>Albania</td>
      <td>5.766000</td>
      <td>16.510579</td>
      <td>7.891086</td>
      <td>14.082553</td>
      <td>11.186264</td>
      <td>9.538675</td>
      <td>6.885189</td>
      <td>8.376691</td>
      <td>8.896982</td>
      <td>4.405134</td>
      <td>7.127992</td>
      <td>5.098855</td>
      <td>100.0</td>
      <td>20.187305</td>
      <td>25.806452</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>4.0</td>
      <td>2002</td>
      <td>12.0</td>
      <td>0.897816</td>
      <td>0.595164</td>
      <td>0.863495</td>
      <td>0.719969</td>
      <td>0.511700</td>
      <td>0.435257</td>
      <td>0.205148</td>
      <td>0.602964</td>
      <td>0.464119</td>
      <td>0.565523</td>
      <td>0.701248</td>
      <td>11.0</td>
      <td>Algeria_2002</td>
      <td>Algeria</td>
      <td>6.562402</td>
      <td>13.681208</td>
      <td>9.069298</td>
      <td>13.158208</td>
      <td>10.971116</td>
      <td>7.797456</td>
      <td>6.632592</td>
      <td>3.126114</td>
      <td>9.188161</td>
      <td>7.072388</td>
      <td>8.617616</td>
      <td>10.685843</td>
      <td>100.0</td>
      <td>22.893142</td>
      <td>19.992868</td>
    </tr>
    <tr>
      <th>Argentina</th>
      <td>1.0</td>
      <td>1984</td>
      <td>32.0</td>
      <td>0.862687</td>
      <td>0.366169</td>
      <td>0.528358</td>
      <td>0.264677</td>
      <td>0.560199</td>
      <td>0.468657</td>
      <td>0.371144</td>
      <td>0.338308</td>
      <td>0.331343</td>
      <td>0.383085</td>
      <td>0.352239</td>
      <td>11.0</td>
      <td>Argentina_1984</td>
      <td>Argentina</td>
      <td>4.826866</td>
      <td>17.872604</td>
      <td>7.586065</td>
      <td>10.946197</td>
      <td>5.483405</td>
      <td>11.605854</td>
      <td>9.709338</td>
      <td>7.689136</td>
      <td>7.008864</td>
      <td>6.864564</td>
      <td>7.936508</td>
      <td>7.297464</td>
      <td>100.0</td>
      <td>23.582766</td>
      <td>26.881055</td>
    </tr>
    <tr>
      <th>Argentina</th>
      <td>2.0</td>
      <td>1991</td>
      <td>32.0</td>
      <td>0.840319</td>
      <td>0.407186</td>
      <td>0.632735</td>
      <td>0.428144</td>
      <td>0.435130</td>
      <td>0.546906</td>
      <td>0.299401</td>
      <td>0.501996</td>
      <td>0.581836</td>
      <td>0.506986</td>
      <td>0.584830</td>
      <td>11.0</td>
      <td>Argentina_1991</td>
      <td>Argentina</td>
      <td>5.765469</td>
      <td>14.575039</td>
      <td>7.062489</td>
      <td>10.974554</td>
      <td>7.426000</td>
      <td>7.547170</td>
      <td>9.485892</td>
      <td>5.193007</td>
      <td>8.706941</td>
      <td>10.091743</td>
      <td>8.793491</td>
      <td>10.143673</td>
      <td>100.0</td>
      <td>28.284577</td>
      <td>19.802666</td>
    </tr>
  </tbody>
</table>
</div>




```python
country_averages.shape
```




    (104, 32)




```python
country_averages['normalized achieving_responsible_initiative'].mean()
```




    25.398708958475307




```python
country_averages['normalized hours_holidays_pressure'].mean()
```




    20.278564639572803




```python
country_averages[normalized_refined_job_qs].mean().sort_values(ascending=False)
```




    normalized Important in a job: good pay                            13.585293
    normalized Important in a job: good job security                   11.864896
    normalized Important in a job: a job that meets one´s abilities    10.681074
    normalized Important in a job: a job that is interesting           10.071307
    normalized Important in a job: that you can achieve something       9.724731
    normalized Important in a job: good hours                           8.396740
    normalized Important in a job: a respected job                      8.120157
    normalized Important in a job: an opportunity to use initiative     7.863789
    normalized Important in a job: a responsible job                    7.810189
    normalized Important in a job: not too much pressure                6.757344
    normalized Important in a job: generous holidays                    5.124481
    dtype: float64



Together, these two features account for ~45% of the variance. "Good pay" and "good job security" are also important, accounting for ~14% and ~12% of responses respectively. In addition, they are correlated (0.32) so we'll try engineering a new feature combining both of these.


```python
# Security_pay

country_averages['normalized security_pay'] = \
    country_averages['normalized Important in a job: good job security'] + \
    country_averages['normalized Important in a job: good pay']
```


```python
country_averages['normalized security_pay'].mean()
```




    25.450188769249337




```python
engineered_job_attributes = ['normalized achieving_responsible_initiative', 
                             'normalized hours_holidays_pressure', 
                             'normalized security_pay']
```

Now, our three engineered features account for ~75% of responses, using only three features vs. the original eleven. While this is not as mathematically rigorous as PCA (in terms of maximizing variance captured), creating meaningful combinations of variables helps us maintain interpretability.

#### Clustering analysis


```python
# Write a plotter function

def plotter(df, input_cols, \
            k_means_n_clusters = 3, agglom_n_clusters = 3, \
            dbscan_eps = 0.5, dbscan_min_samples = 5, n_rows = 1, n_cols = 3, _cmap = plt.get_cmap('tab20')):
    '''function to visualize kmean, agglomerative and dbscan clustering
    args: dataframe, input_cols --> to pull in from df, 
    kwargs: k_means_n_clusters, agglom_n_clusters, dbscan_eps, dbscan_min_samples, n_rows, n_cols --> rows/ cols of subplots, _cmap'''
    # pull out columns to plot and scale
    cols_to_plot = df[input_cols]
    figure, ax = plt.subplots(nrows = n_rows, ncols=n_cols, figsize=(18,5))
    # k means
    kmeans = KMeans(n_clusters=k_means_n_clusters)
    kmeans.fit(cols_to_plot)
    df['kmeans_labels'] = kmeans.labels_
    for index, sub_dataframe in df.groupby('kmeans_labels'):
        ax[0].set_title("K means")
        ax[0].scatter(sub_dataframe[input_cols[0]], sub_dataframe[input_cols[1]], c=_cmap.colors[index], label = f"Index {index}")
        ax[0].set_xlabel(input_cols[0])
        ax[0].set_ylabel(input_cols[1])
        ax[0].legend()
    # agglom
    agglom = AgglomerativeClustering(n_clusters=agglom_n_clusters)
    agglom.fit(cols_to_plot)
    df['agglom_labels'] = agglom.labels_
    for index, sub_dataframe in df.groupby('agglom_labels'):
        ax[1].set_title("Agglomerative clustering")
        ax[1].scatter(sub_dataframe[input_cols[0]], sub_dataframe[input_cols[1]], c=_cmap.colors[index], label = f"Index {index}")
        ax[1].set_xlabel(input_cols[0])
        ax[1].set_ylabel(input_cols[1])
        ax[1].legend()
    # dbscan
    dbmodel = DBSCAN(eps = dbscan_eps, min_samples=dbscan_min_samples)
    dbmodel.fit(cols_to_plot)
    df['dbscan_labels'] = dbmodel.labels_
    for index, sub_dataframe in df.groupby('dbscan_labels'):
        ax[2].set_title("DBSCAN")
        ax[2].scatter(sub_dataframe[input_cols[0]], sub_dataframe[input_cols[1]], c=_cmap.colors[index], label = f"Index {index}")
        ax[2].set_xlabel(input_cols[0])
        ax[2].set_ylabel(input_cols[1])
        ax[2].legend()
```


```python
# Plot countries, using different clustering algorithms and variables
plotter(df = country_averages, 
        input_cols = ['normalized achieving_responsible_initiative', 'normalized hours_holidays_pressure'],
        k_means_n_clusters = 3, agglom_n_clusters = 3, dbscan_eps = 2)
```


![png](/images/World_Values--Full_dataset_files/World_Values--Full_dataset_68_0.png)



```python
plotter(df = country_averages, 
        input_cols = ['normalized achieving_responsible_initiative', 'normalized security_pay'],
        k_means_n_clusters = 3, agglom_n_clusters = 3, dbscan_eps = 2)
```


![png](/images/World_Values--Full_dataset_files/World_Values--Full_dataset_69_0.png)



```python
plotter(df = country_averages, 
        input_cols = ['normalized hours_holidays_pressure', 'normalized security_pay'],
        k_means_n_clusters = 3, agglom_n_clusters = 3, dbscan_eps = 2)
```


![png](/images/World_Values--Full_dataset_files/World_Values--Full_dataset_70_0.png)


#### The top left chart (kmeans clustering with the two axes as achieving/ responsible/ initiative and hours/ holidays/ pressure accounts) provides the best visual separation. Proceeding with this...


```python
# Modify our plotter to focus on k-means and add point labels

def k_means_plotter(df, input_cols, k_means_n_clusters = 3, n_rows = 1, n_cols = 1, _cmap = plt.get_cmap('tab20')):
    '''function to visualize kmean clustering
    args: dataframe, input_cols --> to pull in from df, agglom_n_clusters, dbscan_eps, dbscan_min_samples
    kwargs: k_means_n_clusters = 3, n_rows = 1, n_cols =1 --> rows/ cols of subplots, _cmap = plt.get_cmap('tab20')'''
    # pull out columns to plot and scale
    cols_to_plot = df[input_cols]
    figure, ax = plt.subplots(nrows = n_rows, ncols=n_cols, figsize=(18,11))
    # k means
    kmeans = KMeans(n_clusters=k_means_n_clusters, random_state=_random_state)
    kmeans.fit(cols_to_plot)
    df['kmeans_labels'] = kmeans.labels_
    for index, sub_dataframe in df.groupby('kmeans_labels'):
        ax.set_title("\n Country value systems\n ", size = 18)
        ax.scatter(sub_dataframe[input_cols[0]], sub_dataframe[input_cols[1]], c=_cmap.colors[index], label = f"Index {index}")
        ax.set_xlabel('% of responses focused on achievement', size = 14)
        ax.tick_params(labelsize = 14)
        #ax.set_xlim(xmax = 33)
        ax.set_ylabel('% of responses focused on lifestyle', size = 14)
        for j, k in sub_dataframe['country_year'].iteritems():
            x = sub_dataframe[input_cols[0]][j]+.1
            y = sub_dataframe[input_cols[1]][j]+.05
            label = k
            ax.annotate(label, (x, y), rotation= 20, ha = 'left', va = 'bottom', size = 10)
        ax.legend(["Balanced","Achievement","Lifestyle"], fontsize = 12)
```


```python
# Reset index to unique values
country_averages.index = country_averages['country_year']
```


```python
k_means_plotter(df = country_averages, 
        input_cols = ['normalized achieving_responsible_initiative', 'normalized hours_holidays_pressure'])
```


![png](/images/World_Values--Full_dataset_files/World_Values--Full_dataset_74_0.png)


#### The clustering analysis identifies three groups:
- **The lifestyle group**: Countries where people are less concerned with having a high-achieving job and moderately concerned about lifestyle (vacations, hours, pressure) - including Russia, Albania and Slovakia
- **The achievement group**: Countries where people want a high-achieving job and care less about lifestyle - including Norway, Sweden and Australia
- **The balanced group**: Countries that balance achievement with lifestyle - including Turkey, Morocco and South Korea

#### Next, we will use multi-class logistic regression to see if we can predict a country's cluster based on economic variables.

#### Two notes:
- The boundaries between the groups are somewhat fuzzy (e.g., the U.S. in 1990 could also be assigned to the high-achievement group
- Many countries (for example Japan and Turkey) are fairly stable in values over time. Building on this analysis, we could look at what drives changes.


```python
# Merge in labels

cluster_labels_df = pd.DataFrame([{'kmeans_labels': 1, 'kmeans_value_cluster' :'achievement'},
                                  {'kmeans_labels': 2, 'kmeans_value_cluster' :'lifestyle'},
                                  {'kmeans_labels': 0, 'kmeans_value_cluster' :'balanced'}])

country_averages = pd.merge(country_averages, cluster_labels_df, on = 'kmeans_labels')
country_averages.index = country_averages['country_year']
```


```python
country_averages.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Wave</th>
      <th>Year survey</th>
      <th>Country/region</th>
      <th>Important in a job: good pay</th>
      <th>Important in a job: not too much pressure</th>
      <th>Important in a job: good job security</th>
      <th>Important in a job: a respected job</th>
      <th>Important in a job: good hours</th>
      <th>Important in a job: an opportunity to use initiative</th>
      <th>Important in a job: generous holidays</th>
      <th>Important in a job: that you can achieve something</th>
      <th>Important in a job: a responsible job</th>
      <th>Important in a job: a job that is interesting</th>
      <th>Important in a job: a job that meets one´s abilities</th>
      <th>number_of_jobs_qs_answered</th>
      <th>country_year</th>
      <th>country_name</th>
      <th>total_mentions</th>
      <th>normalized Important in a job: good pay</th>
      <th>normalized Important in a job: not too much pressure</th>
      <th>normalized Important in a job: good job security</th>
      <th>normalized Important in a job: a respected job</th>
      <th>normalized Important in a job: good hours</th>
      <th>normalized Important in a job: an opportunity to use initiative</th>
      <th>normalized Important in a job: generous holidays</th>
      <th>normalized Important in a job: that you can achieve something</th>
      <th>normalized Important in a job: a responsible job</th>
      <th>normalized Important in a job: a job that is interesting</th>
      <th>normalized Important in a job: a job that meets one´s abilities</th>
      <th>normalized total_mentions</th>
      <th>normalized achieving_responsible_initiative</th>
      <th>normalized hours_holidays_pressure</th>
      <th>normalized security_pay</th>
      <th>kmeans_labels</th>
      <th>agglom_labels</th>
      <th>dbscan_labels</th>
      <th>kmeans_value_cluster</th>
    </tr>
    <tr>
      <th>country_year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Albania_1998</th>
      <td>3.0</td>
      <td>1998</td>
      <td>8.0</td>
      <td>0.994995</td>
      <td>0.372372</td>
      <td>0.853854</td>
      <td>0.417417</td>
      <td>0.610611</td>
      <td>0.426426</td>
      <td>0.533534</td>
      <td>0.521522</td>
      <td>0.173173</td>
      <td>0.458458</td>
      <td>0.888889</td>
      <td>11.0</td>
      <td>Albania_1998</td>
      <td>Albania</td>
      <td>6.251251</td>
      <td>15.916733</td>
      <td>5.956765</td>
      <td>13.658927</td>
      <td>6.677342</td>
      <td>9.767814</td>
      <td>6.821457</td>
      <td>8.534828</td>
      <td>8.342674</td>
      <td>2.770216</td>
      <td>7.333867</td>
      <td>14.219376</td>
      <td>100.0</td>
      <td>17.934347</td>
      <td>24.259408</td>
      <td>29.575661</td>
      <td>2</td>
      <td>0</td>
      <td>-1</td>
      <td>lifestyle</td>
    </tr>
    <tr>
      <th>Albania_2002</th>
      <td>4.0</td>
      <td>2002</td>
      <td>8.0</td>
      <td>0.952000</td>
      <td>0.455000</td>
      <td>0.812000</td>
      <td>0.645000</td>
      <td>0.550000</td>
      <td>0.397000</td>
      <td>0.483000</td>
      <td>0.513000</td>
      <td>0.254000</td>
      <td>0.411000</td>
      <td>0.294000</td>
      <td>11.0</td>
      <td>Albania_2002</td>
      <td>Albania</td>
      <td>5.766000</td>
      <td>16.510579</td>
      <td>7.891086</td>
      <td>14.082553</td>
      <td>11.186264</td>
      <td>9.538675</td>
      <td>6.885189</td>
      <td>8.376691</td>
      <td>8.896982</td>
      <td>4.405134</td>
      <td>7.127992</td>
      <td>5.098855</td>
      <td>100.0</td>
      <td>20.187305</td>
      <td>25.806452</td>
      <td>30.593132</td>
      <td>2</td>
      <td>0</td>
      <td>-1</td>
      <td>lifestyle</td>
    </tr>
    <tr>
      <th>Algeria_2002</th>
      <td>4.0</td>
      <td>2002</td>
      <td>12.0</td>
      <td>0.897816</td>
      <td>0.595164</td>
      <td>0.863495</td>
      <td>0.719969</td>
      <td>0.511700</td>
      <td>0.435257</td>
      <td>0.205148</td>
      <td>0.602964</td>
      <td>0.464119</td>
      <td>0.565523</td>
      <td>0.701248</td>
      <td>11.0</td>
      <td>Algeria_2002</td>
      <td>Algeria</td>
      <td>6.562402</td>
      <td>13.681208</td>
      <td>9.069298</td>
      <td>13.158208</td>
      <td>10.971116</td>
      <td>7.797456</td>
      <td>6.632592</td>
      <td>3.126114</td>
      <td>9.188161</td>
      <td>7.072388</td>
      <td>8.617616</td>
      <td>10.685843</td>
      <td>100.0</td>
      <td>22.893142</td>
      <td>19.992868</td>
      <td>26.839415</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>lifestyle</td>
    </tr>
    <tr>
      <th>Armenia_1997</th>
      <td>3.0</td>
      <td>1997</td>
      <td>51.0</td>
      <td>0.935000</td>
      <td>0.530000</td>
      <td>0.773000</td>
      <td>0.698000</td>
      <td>0.487000</td>
      <td>0.456500</td>
      <td>0.306500</td>
      <td>0.655500</td>
      <td>0.427500</td>
      <td>0.773500</td>
      <td>0.662000</td>
      <td>11.0</td>
      <td>Armenia_1997</td>
      <td>Armenia</td>
      <td>6.704500</td>
      <td>13.945857</td>
      <td>7.905138</td>
      <td>11.529570</td>
      <td>10.410918</td>
      <td>7.263778</td>
      <td>6.808860</td>
      <td>4.571556</td>
      <td>9.777015</td>
      <td>6.376314</td>
      <td>11.537027</td>
      <td>9.873965</td>
      <td>100.0</td>
      <td>22.962190</td>
      <td>19.740473</td>
      <td>25.475427</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>lifestyle</td>
    </tr>
    <tr>
      <th>Azerbaijan_1997</th>
      <td>3.0</td>
      <td>1997</td>
      <td>31.0</td>
      <td>0.952048</td>
      <td>0.557443</td>
      <td>0.742757</td>
      <td>0.648851</td>
      <td>0.546953</td>
      <td>0.351648</td>
      <td>0.383616</td>
      <td>0.490010</td>
      <td>0.446553</td>
      <td>0.812687</td>
      <td>0.734266</td>
      <td>11.0</td>
      <td>Azerbaijan_1997</td>
      <td>Azerbaijan</td>
      <td>6.666833</td>
      <td>14.280363</td>
      <td>8.361430</td>
      <td>11.141080</td>
      <td>9.732524</td>
      <td>8.204091</td>
      <td>5.274594</td>
      <td>5.754102</td>
      <td>7.349966</td>
      <td>6.698134</td>
      <td>12.190005</td>
      <td>11.013711</td>
      <td>100.0</td>
      <td>19.322694</td>
      <td>22.319622</td>
      <td>25.421443</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>lifestyle</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Confirm value clusters correctly mapped by looking at one example from each cluster
balanced_check = country_averages.loc['Uganda_2001','kmeans_value_cluster'] # balanced cluster
achievement_check = country_averages.loc['Peru_1996','kmeans_value_cluster'] # achievement cluster
lifestyle_check = country_averages.loc['Finland_1981','kmeans_value_cluster'] # lifestyle cluster

balanced_check, achievement_check, lifestyle_check
```




    ('balanced', 'achievement', 'lifestyle')




```python
country_averages.shape
```




    (104, 37)



### Logistic regression

#### Merge in indicator data - GDP


```python
# Read in dataframe (from https://data.worldbank.org/, accessed July 2018)
gdp_df = pd.read_csv('./indicator_data/World_bank_GDP_per_cap_PPP_df.csv')
```


```python
gdp_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country Name</th>
      <th>1990</th>
      <th>1991</th>
      <th>1992</th>
      <th>1993</th>
      <th>1994</th>
      <th>1995</th>
      <th>1996</th>
      <th>1997</th>
      <th>1998</th>
      <th>1999</th>
      <th>2000</th>
      <th>2001</th>
      <th>2002</th>
      <th>2003</th>
      <th>2004</th>
      <th>2005</th>
      <th>2006</th>
      <th>2007</th>
      <th>2008</th>
      <th>2009</th>
      <th>2010</th>
      <th>2011</th>
      <th>2012</th>
      <th>2013</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
      <th>2017</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aruba</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>35973.780510</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>875.517610</td>
      <td>922.829449</td>
      <td>916.334475</td>
      <td>1011.595524</td>
      <td>1065.619665</td>
      <td>1210.479265</td>
      <td>1247.066144</td>
      <td>1482.098837</td>
      <td>1581.600836</td>
      <td>1660.739856</td>
      <td>1873.153946</td>
      <td>1913.160644</td>
      <td>1937.235365</td>
      <td>1926.357336</td>
      <td>1944.117005</td>
      <td>1980.516177</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Angola</td>
      <td>2217.438380</td>
      <td>2243.546096</td>
      <td>2091.590164</td>
      <td>1574.909779</td>
      <td>1578.440193</td>
      <td>1797.522525</td>
      <td>2019.744617</td>
      <td>2144.35083</td>
      <td>2208.517559</td>
      <td>2227.240941</td>
      <td>2277.374611</td>
      <td>2349.731596</td>
      <td>2620.884723</td>
      <td>2698.044285</td>
      <td>2948.770248</td>
      <td>3550.720284</td>
      <td>4202.208484</td>
      <td>5128.353450</td>
      <td>5743.098881</td>
      <td>5718.685523</td>
      <td>5778.380803</td>
      <td>5911.254092</td>
      <td>6110.423392</td>
      <td>6403.767556</td>
      <td>6591.538820</td>
      <td>6631.618694</td>
      <td>6440.990242</td>
      <td>6388.960022</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albania</td>
      <td>2722.280344</td>
      <td>1992.560528</td>
      <td>1902.751280</td>
      <td>2148.099954</td>
      <td>2390.531021</td>
      <td>2782.093734</td>
      <td>3109.943347</td>
      <td>2838.04492</td>
      <td>3209.615372</td>
      <td>3690.682981</td>
      <td>4029.016971</td>
      <td>4457.111638</td>
      <td>4754.653697</td>
      <td>5114.721695</td>
      <td>5522.982134</td>
      <td>5942.884009</td>
      <td>6631.835687</td>
      <td>7291.239751</td>
      <td>8228.337106</td>
      <td>8813.252003</td>
      <td>9637.345368</td>
      <td>10207.764700</td>
      <td>10526.241220</td>
      <td>10570.953000</td>
      <td>11259.264500</td>
      <td>11334.220930</td>
      <td>11559.300840</td>
      <td>12020.690730</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Andorra</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check shape of dataframe
gdp_df.shape
```




    (264, 29)




```python
# Check percent nulls
print(f'{gdp_df.isnull().sum().sum()*100/(gdp_df.shape[0]*gdp_df.shape[1]):.0f}% null')
```

    13% null



```python
# The country_averages df has a two-level index (country_year). We need to stack our gdp dataframe to match
```


```python
gdp_df.index = gdp_df['Country Name']
gdp_df.drop('Country Name', axis = 1, inplace = True)

gdp_df = pd.DataFrame(gdp_df.stack(), columns=["GDP per capita, PPP"])

gdp_df.reset_index(inplace=True)

gdp_df.index = gdp_df['Country Name'] + "_" + gdp_df['level_1']

gdp_df.drop(['Country Name', 'level_1'], axis = 1, inplace= True)

gdp_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GDP per capita, PPP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Aruba_2011</th>
      <td>35973.780510</td>
    </tr>
    <tr>
      <th>Afghanistan_2002</th>
      <td>875.517610</td>
    </tr>
    <tr>
      <th>Afghanistan_2003</th>
      <td>922.829449</td>
    </tr>
    <tr>
      <th>Afghanistan_2004</th>
      <td>916.334475</td>
    </tr>
    <tr>
      <th>Afghanistan_2005</th>
      <td>1011.595524</td>
    </tr>
  </tbody>
</table>
</div>




```python
country_averages.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Wave</th>
      <th>Year survey</th>
      <th>Country/region</th>
      <th>Important in a job: good pay</th>
      <th>Important in a job: not too much pressure</th>
      <th>Important in a job: good job security</th>
      <th>Important in a job: a respected job</th>
      <th>Important in a job: good hours</th>
      <th>Important in a job: an opportunity to use initiative</th>
      <th>Important in a job: generous holidays</th>
      <th>Important in a job: that you can achieve something</th>
      <th>Important in a job: a responsible job</th>
      <th>Important in a job: a job that is interesting</th>
      <th>Important in a job: a job that meets one´s abilities</th>
      <th>number_of_jobs_qs_answered</th>
      <th>country_year</th>
      <th>country_name</th>
      <th>total_mentions</th>
      <th>normalized Important in a job: good pay</th>
      <th>normalized Important in a job: not too much pressure</th>
      <th>normalized Important in a job: good job security</th>
      <th>normalized Important in a job: a respected job</th>
      <th>normalized Important in a job: good hours</th>
      <th>normalized Important in a job: an opportunity to use initiative</th>
      <th>normalized Important in a job: generous holidays</th>
      <th>normalized Important in a job: that you can achieve something</th>
      <th>normalized Important in a job: a responsible job</th>
      <th>normalized Important in a job: a job that is interesting</th>
      <th>normalized Important in a job: a job that meets one´s abilities</th>
      <th>normalized total_mentions</th>
      <th>normalized achieving_responsible_initiative</th>
      <th>normalized hours_holidays_pressure</th>
      <th>normalized security_pay</th>
      <th>kmeans_labels</th>
      <th>agglom_labels</th>
      <th>dbscan_labels</th>
      <th>kmeans_value_cluster</th>
    </tr>
    <tr>
      <th>country_year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Albania_1998</th>
      <td>3.0</td>
      <td>1998</td>
      <td>8.0</td>
      <td>0.994995</td>
      <td>0.372372</td>
      <td>0.853854</td>
      <td>0.417417</td>
      <td>0.610611</td>
      <td>0.426426</td>
      <td>0.533534</td>
      <td>0.521522</td>
      <td>0.173173</td>
      <td>0.458458</td>
      <td>0.888889</td>
      <td>11.0</td>
      <td>Albania_1998</td>
      <td>Albania</td>
      <td>6.251251</td>
      <td>15.916733</td>
      <td>5.956765</td>
      <td>13.658927</td>
      <td>6.677342</td>
      <td>9.767814</td>
      <td>6.821457</td>
      <td>8.534828</td>
      <td>8.342674</td>
      <td>2.770216</td>
      <td>7.333867</td>
      <td>14.219376</td>
      <td>100.0</td>
      <td>17.934347</td>
      <td>24.259408</td>
      <td>29.575661</td>
      <td>2</td>
      <td>0</td>
      <td>-1</td>
      <td>lifestyle</td>
    </tr>
    <tr>
      <th>Albania_2002</th>
      <td>4.0</td>
      <td>2002</td>
      <td>8.0</td>
      <td>0.952000</td>
      <td>0.455000</td>
      <td>0.812000</td>
      <td>0.645000</td>
      <td>0.550000</td>
      <td>0.397000</td>
      <td>0.483000</td>
      <td>0.513000</td>
      <td>0.254000</td>
      <td>0.411000</td>
      <td>0.294000</td>
      <td>11.0</td>
      <td>Albania_2002</td>
      <td>Albania</td>
      <td>5.766000</td>
      <td>16.510579</td>
      <td>7.891086</td>
      <td>14.082553</td>
      <td>11.186264</td>
      <td>9.538675</td>
      <td>6.885189</td>
      <td>8.376691</td>
      <td>8.896982</td>
      <td>4.405134</td>
      <td>7.127992</td>
      <td>5.098855</td>
      <td>100.0</td>
      <td>20.187305</td>
      <td>25.806452</td>
      <td>30.593132</td>
      <td>2</td>
      <td>0</td>
      <td>-1</td>
      <td>lifestyle</td>
    </tr>
    <tr>
      <th>Algeria_2002</th>
      <td>4.0</td>
      <td>2002</td>
      <td>12.0</td>
      <td>0.897816</td>
      <td>0.595164</td>
      <td>0.863495</td>
      <td>0.719969</td>
      <td>0.511700</td>
      <td>0.435257</td>
      <td>0.205148</td>
      <td>0.602964</td>
      <td>0.464119</td>
      <td>0.565523</td>
      <td>0.701248</td>
      <td>11.0</td>
      <td>Algeria_2002</td>
      <td>Algeria</td>
      <td>6.562402</td>
      <td>13.681208</td>
      <td>9.069298</td>
      <td>13.158208</td>
      <td>10.971116</td>
      <td>7.797456</td>
      <td>6.632592</td>
      <td>3.126114</td>
      <td>9.188161</td>
      <td>7.072388</td>
      <td>8.617616</td>
      <td>10.685843</td>
      <td>100.0</td>
      <td>22.893142</td>
      <td>19.992868</td>
      <td>26.839415</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>lifestyle</td>
    </tr>
    <tr>
      <th>Armenia_1997</th>
      <td>3.0</td>
      <td>1997</td>
      <td>51.0</td>
      <td>0.935000</td>
      <td>0.530000</td>
      <td>0.773000</td>
      <td>0.698000</td>
      <td>0.487000</td>
      <td>0.456500</td>
      <td>0.306500</td>
      <td>0.655500</td>
      <td>0.427500</td>
      <td>0.773500</td>
      <td>0.662000</td>
      <td>11.0</td>
      <td>Armenia_1997</td>
      <td>Armenia</td>
      <td>6.704500</td>
      <td>13.945857</td>
      <td>7.905138</td>
      <td>11.529570</td>
      <td>10.410918</td>
      <td>7.263778</td>
      <td>6.808860</td>
      <td>4.571556</td>
      <td>9.777015</td>
      <td>6.376314</td>
      <td>11.537027</td>
      <td>9.873965</td>
      <td>100.0</td>
      <td>22.962190</td>
      <td>19.740473</td>
      <td>25.475427</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>lifestyle</td>
    </tr>
    <tr>
      <th>Azerbaijan_1997</th>
      <td>3.0</td>
      <td>1997</td>
      <td>31.0</td>
      <td>0.952048</td>
      <td>0.557443</td>
      <td>0.742757</td>
      <td>0.648851</td>
      <td>0.546953</td>
      <td>0.351648</td>
      <td>0.383616</td>
      <td>0.490010</td>
      <td>0.446553</td>
      <td>0.812687</td>
      <td>0.734266</td>
      <td>11.0</td>
      <td>Azerbaijan_1997</td>
      <td>Azerbaijan</td>
      <td>6.666833</td>
      <td>14.280363</td>
      <td>8.361430</td>
      <td>11.141080</td>
      <td>9.732524</td>
      <td>8.204091</td>
      <td>5.274594</td>
      <td>5.754102</td>
      <td>7.349966</td>
      <td>6.698134</td>
      <td>12.190005</td>
      <td>11.013711</td>
      <td>100.0</td>
      <td>19.322694</td>
      <td>22.319622</td>
      <td>25.421443</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>lifestyle</td>
    </tr>
  </tbody>
</table>
</div>




```python
country_averages.shape
```




    (104, 37)




```python
# Check our country_averages df has no nulls
assert country_averages.isnull().sum().sum() == 0
```


```python
# Merge in GDP information
```


```python
job_and_gdp_df = pd.merge(country_averages, gdp_df, how = 'left', left_index = True, right_index= True)
```


```python
# Preview new dataframe
```


```python
job_and_gdp_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Wave</th>
      <th>Year survey</th>
      <th>Country/region</th>
      <th>Important in a job: good pay</th>
      <th>Important in a job: not too much pressure</th>
      <th>Important in a job: good job security</th>
      <th>Important in a job: a respected job</th>
      <th>Important in a job: good hours</th>
      <th>Important in a job: an opportunity to use initiative</th>
      <th>Important in a job: generous holidays</th>
      <th>Important in a job: that you can achieve something</th>
      <th>Important in a job: a responsible job</th>
      <th>Important in a job: a job that is interesting</th>
      <th>Important in a job: a job that meets one´s abilities</th>
      <th>number_of_jobs_qs_answered</th>
      <th>country_year</th>
      <th>country_name</th>
      <th>total_mentions</th>
      <th>normalized Important in a job: good pay</th>
      <th>normalized Important in a job: not too much pressure</th>
      <th>normalized Important in a job: good job security</th>
      <th>normalized Important in a job: a respected job</th>
      <th>normalized Important in a job: good hours</th>
      <th>normalized Important in a job: an opportunity to use initiative</th>
      <th>normalized Important in a job: generous holidays</th>
      <th>normalized Important in a job: that you can achieve something</th>
      <th>normalized Important in a job: a responsible job</th>
      <th>normalized Important in a job: a job that is interesting</th>
      <th>normalized Important in a job: a job that meets one´s abilities</th>
      <th>normalized total_mentions</th>
      <th>normalized achieving_responsible_initiative</th>
      <th>normalized hours_holidays_pressure</th>
      <th>normalized security_pay</th>
      <th>kmeans_labels</th>
      <th>agglom_labels</th>
      <th>dbscan_labels</th>
      <th>kmeans_value_cluster</th>
      <th>GDP per capita, PPP</th>
    </tr>
    <tr>
      <th>country_year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Albania_1998</th>
      <td>3.0</td>
      <td>1998</td>
      <td>8.0</td>
      <td>0.994995</td>
      <td>0.372372</td>
      <td>0.853854</td>
      <td>0.417417</td>
      <td>0.610611</td>
      <td>0.426426</td>
      <td>0.533534</td>
      <td>0.521522</td>
      <td>0.173173</td>
      <td>0.458458</td>
      <td>0.888889</td>
      <td>11.0</td>
      <td>Albania_1998</td>
      <td>Albania</td>
      <td>6.251251</td>
      <td>15.916733</td>
      <td>5.956765</td>
      <td>13.658927</td>
      <td>6.677342</td>
      <td>9.767814</td>
      <td>6.821457</td>
      <td>8.534828</td>
      <td>8.342674</td>
      <td>2.770216</td>
      <td>7.333867</td>
      <td>14.219376</td>
      <td>100.0</td>
      <td>17.934347</td>
      <td>24.259408</td>
      <td>29.575661</td>
      <td>2</td>
      <td>0</td>
      <td>-1</td>
      <td>lifestyle</td>
      <td>3209.615372</td>
    </tr>
    <tr>
      <th>Albania_2002</th>
      <td>4.0</td>
      <td>2002</td>
      <td>8.0</td>
      <td>0.952000</td>
      <td>0.455000</td>
      <td>0.812000</td>
      <td>0.645000</td>
      <td>0.550000</td>
      <td>0.397000</td>
      <td>0.483000</td>
      <td>0.513000</td>
      <td>0.254000</td>
      <td>0.411000</td>
      <td>0.294000</td>
      <td>11.0</td>
      <td>Albania_2002</td>
      <td>Albania</td>
      <td>5.766000</td>
      <td>16.510579</td>
      <td>7.891086</td>
      <td>14.082553</td>
      <td>11.186264</td>
      <td>9.538675</td>
      <td>6.885189</td>
      <td>8.376691</td>
      <td>8.896982</td>
      <td>4.405134</td>
      <td>7.127992</td>
      <td>5.098855</td>
      <td>100.0</td>
      <td>20.187305</td>
      <td>25.806452</td>
      <td>30.593132</td>
      <td>2</td>
      <td>0</td>
      <td>-1</td>
      <td>lifestyle</td>
      <td>4754.653697</td>
    </tr>
    <tr>
      <th>Algeria_2002</th>
      <td>4.0</td>
      <td>2002</td>
      <td>12.0</td>
      <td>0.897816</td>
      <td>0.595164</td>
      <td>0.863495</td>
      <td>0.719969</td>
      <td>0.511700</td>
      <td>0.435257</td>
      <td>0.205148</td>
      <td>0.602964</td>
      <td>0.464119</td>
      <td>0.565523</td>
      <td>0.701248</td>
      <td>11.0</td>
      <td>Algeria_2002</td>
      <td>Algeria</td>
      <td>6.562402</td>
      <td>13.681208</td>
      <td>9.069298</td>
      <td>13.158208</td>
      <td>10.971116</td>
      <td>7.797456</td>
      <td>6.632592</td>
      <td>3.126114</td>
      <td>9.188161</td>
      <td>7.072388</td>
      <td>8.617616</td>
      <td>10.685843</td>
      <td>100.0</td>
      <td>22.893142</td>
      <td>19.992868</td>
      <td>26.839415</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>lifestyle</td>
      <td>8911.680680</td>
    </tr>
    <tr>
      <th>Armenia_1997</th>
      <td>3.0</td>
      <td>1997</td>
      <td>51.0</td>
      <td>0.935000</td>
      <td>0.530000</td>
      <td>0.773000</td>
      <td>0.698000</td>
      <td>0.487000</td>
      <td>0.456500</td>
      <td>0.306500</td>
      <td>0.655500</td>
      <td>0.427500</td>
      <td>0.773500</td>
      <td>0.662000</td>
      <td>11.0</td>
      <td>Armenia_1997</td>
      <td>Armenia</td>
      <td>6.704500</td>
      <td>13.945857</td>
      <td>7.905138</td>
      <td>11.529570</td>
      <td>10.410918</td>
      <td>7.263778</td>
      <td>6.808860</td>
      <td>4.571556</td>
      <td>9.777015</td>
      <td>6.376314</td>
      <td>11.537027</td>
      <td>9.873965</td>
      <td>100.0</td>
      <td>22.962190</td>
      <td>19.740473</td>
      <td>25.475427</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>lifestyle</td>
      <td>1843.378545</td>
    </tr>
    <tr>
      <th>Azerbaijan_1997</th>
      <td>3.0</td>
      <td>1997</td>
      <td>31.0</td>
      <td>0.952048</td>
      <td>0.557443</td>
      <td>0.742757</td>
      <td>0.648851</td>
      <td>0.546953</td>
      <td>0.351648</td>
      <td>0.383616</td>
      <td>0.490010</td>
      <td>0.446553</td>
      <td>0.812687</td>
      <td>0.734266</td>
      <td>11.0</td>
      <td>Azerbaijan_1997</td>
      <td>Azerbaijan</td>
      <td>6.666833</td>
      <td>14.280363</td>
      <td>8.361430</td>
      <td>11.141080</td>
      <td>9.732524</td>
      <td>8.204091</td>
      <td>5.274594</td>
      <td>5.754102</td>
      <td>7.349966</td>
      <td>6.698134</td>
      <td>12.190005</td>
      <td>11.013711</td>
      <td>100.0</td>
      <td>19.322694</td>
      <td>22.319622</td>
      <td>25.421443</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>lifestyle</td>
      <td>2634.158631</td>
    </tr>
  </tbody>
</table>
</div>



**Assess introduction of null values**

We had introduced 30 nulls, out of 104 values. Refering to the null values using "job_and_gdp_df.sort_values('GDP per capita, PPP', na_position='first').head(15)" we can fix many of these by matching our country names to country names in the World Bank database (e.g., Dominican Republic vs. Dominican Rep.). Post-processing, 11 nulls remain. These are mostly due to timing, as the World Values Survey was administered in a few countries before 1990 whereas the World Bank PPP GDP figures begin in 1990.


```python
job_and_gdp_df['Wave'].count(), job_and_gdp_df['GDP per capita, PPP'].isnull().sum()
```




    (104, 11)



We will drop the nulls from our dataframe going forwards. For easy reference, we will save the dataframe with nulls as job_and_gdp_df_original.


```python
job_and_gdp_df_original = job_and_gdp_df
job_and_gdp_df = job_and_gdp_df.dropna()
```


```python
job_and_gdp_df.shape, job_and_gdp_df_original.shape
```




    ((93, 38), (104, 38))



#### Merge in indicator data - GINI


```python
# Read in dataframe (from https://data.worldbank.org/, accessed July 2018)
gini_df = pd.read_csv('./indicator_data/gini.csv')
```


```python
# Preview data
gini_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country Name</th>
      <th>1980</th>
      <th>1981</th>
      <th>1982</th>
      <th>1983</th>
      <th>1984</th>
      <th>1985</th>
      <th>1986</th>
      <th>1987</th>
      <th>1988</th>
      <th>1989</th>
      <th>1990</th>
      <th>1991</th>
      <th>1992</th>
      <th>1993</th>
      <th>1994</th>
      <th>1995</th>
      <th>1996</th>
      <th>1997</th>
      <th>1998</th>
      <th>1999</th>
      <th>2000</th>
      <th>2001</th>
      <th>2002</th>
      <th>2003</th>
      <th>2004</th>
      <th>2005</th>
      <th>2006</th>
      <th>2007</th>
      <th>2008</th>
      <th>2009</th>
      <th>2010</th>
      <th>2011</th>
      <th>2012</th>
      <th>2013</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
      <th>2017</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aruba</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Angola</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>42.7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albania</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>27.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>31.7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Andorra</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Impute nulls
gini_df.index = gini_df['Country Name']
gini_df.drop('Country Name', axis = 1, inplace=True)
gini_df.fillna(method = 'ffill', axis=1, inplace=True)         # first try a forward fill (i.e., base on the last available value)
gini_df.fillna(method = 'backfill', axis=1, inplace= True)     # next try a backward fill (i.e., base on the next available value)
gini_df.fillna(value = gini_df.mean(), inplace=True)           # where no data is available fill with the dataframe mean
```


```python
gini_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1980</th>
      <th>1981</th>
      <th>1982</th>
      <th>1983</th>
      <th>1984</th>
      <th>1985</th>
      <th>1986</th>
      <th>1987</th>
      <th>1988</th>
      <th>1989</th>
      <th>1990</th>
      <th>1991</th>
      <th>1992</th>
      <th>1993</th>
      <th>1994</th>
      <th>1995</th>
      <th>1996</th>
      <th>1997</th>
      <th>1998</th>
      <th>1999</th>
      <th>2000</th>
      <th>2001</th>
      <th>2002</th>
      <th>2003</th>
      <th>2004</th>
      <th>2005</th>
      <th>2006</th>
      <th>2007</th>
      <th>2008</th>
      <th>2009</th>
      <th>2010</th>
      <th>2011</th>
      <th>2012</th>
      <th>2013</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
      <th>2017</th>
    </tr>
    <tr>
      <th>Country Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Aruba</th>
      <td>40.364024</td>
      <td>40.364024</td>
      <td>40.366463</td>
      <td>40.370122</td>
      <td>40.366463</td>
      <td>40.362805</td>
      <td>40.267073</td>
      <td>40.27622</td>
      <td>40.270122</td>
      <td>40.348171</td>
      <td>40.268293</td>
      <td>40.275</td>
      <td>40.279268</td>
      <td>40.335366</td>
      <td>40.267683</td>
      <td>40.365854</td>
      <td>40.348171</td>
      <td>40.393902</td>
      <td>40.511585</td>
      <td>40.503049</td>
      <td>40.329268</td>
      <td>40.303049</td>
      <td>40.070732</td>
      <td>39.816463</td>
      <td>39.771951</td>
      <td>39.877439</td>
      <td>39.55061</td>
      <td>39.32378</td>
      <td>39.206707</td>
      <td>38.87439</td>
      <td>38.914024</td>
      <td>38.753049</td>
      <td>38.712195</td>
      <td>38.596951</td>
      <td>38.492683</td>
      <td>38.453659</td>
      <td>38.418293</td>
      <td>38.418293</td>
    </tr>
    <tr>
      <th>Afghanistan</th>
      <td>40.364024</td>
      <td>40.364024</td>
      <td>40.366463</td>
      <td>40.370122</td>
      <td>40.366463</td>
      <td>40.362805</td>
      <td>40.267073</td>
      <td>40.27622</td>
      <td>40.270122</td>
      <td>40.348171</td>
      <td>40.268293</td>
      <td>40.275</td>
      <td>40.279268</td>
      <td>40.335366</td>
      <td>40.267683</td>
      <td>40.365854</td>
      <td>40.348171</td>
      <td>40.393902</td>
      <td>40.511585</td>
      <td>40.503049</td>
      <td>40.329268</td>
      <td>40.303049</td>
      <td>40.070732</td>
      <td>39.816463</td>
      <td>39.771951</td>
      <td>39.877439</td>
      <td>39.55061</td>
      <td>39.32378</td>
      <td>39.206707</td>
      <td>38.87439</td>
      <td>38.914024</td>
      <td>38.753049</td>
      <td>38.712195</td>
      <td>38.596951</td>
      <td>38.492683</td>
      <td>38.453659</td>
      <td>38.418293</td>
      <td>38.418293</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>42.700000</td>
      <td>42.700000</td>
      <td>42.700000</td>
      <td>42.700000</td>
      <td>42.700000</td>
      <td>42.700000</td>
      <td>42.700000</td>
      <td>42.70000</td>
      <td>42.700000</td>
      <td>42.700000</td>
      <td>42.700000</td>
      <td>42.700</td>
      <td>42.700000</td>
      <td>42.700000</td>
      <td>42.700000</td>
      <td>42.700000</td>
      <td>42.700000</td>
      <td>42.700000</td>
      <td>42.700000</td>
      <td>42.700000</td>
      <td>42.700000</td>
      <td>42.700000</td>
      <td>42.700000</td>
      <td>42.700000</td>
      <td>42.700000</td>
      <td>42.700000</td>
      <td>42.70000</td>
      <td>42.70000</td>
      <td>42.700000</td>
      <td>42.70000</td>
      <td>42.700000</td>
      <td>42.700000</td>
      <td>42.700000</td>
      <td>42.700000</td>
      <td>42.700000</td>
      <td>42.700000</td>
      <td>42.700000</td>
      <td>42.700000</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.00000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>31.700000</td>
      <td>31.700000</td>
      <td>31.700000</td>
      <td>30.600000</td>
      <td>30.60000</td>
      <td>30.60000</td>
      <td>30.000000</td>
      <td>30.00000</td>
      <td>30.000000</td>
      <td>30.000000</td>
      <td>29.000000</td>
      <td>29.000000</td>
      <td>29.000000</td>
      <td>29.000000</td>
      <td>29.000000</td>
      <td>29.000000</td>
    </tr>
    <tr>
      <th>Andorra</th>
      <td>40.364024</td>
      <td>40.364024</td>
      <td>40.366463</td>
      <td>40.370122</td>
      <td>40.366463</td>
      <td>40.362805</td>
      <td>40.267073</td>
      <td>40.27622</td>
      <td>40.270122</td>
      <td>40.348171</td>
      <td>40.268293</td>
      <td>40.275</td>
      <td>40.279268</td>
      <td>40.335366</td>
      <td>40.267683</td>
      <td>40.365854</td>
      <td>40.348171</td>
      <td>40.393902</td>
      <td>40.511585</td>
      <td>40.503049</td>
      <td>40.329268</td>
      <td>40.303049</td>
      <td>40.070732</td>
      <td>39.816463</td>
      <td>39.771951</td>
      <td>39.877439</td>
      <td>39.55061</td>
      <td>39.32378</td>
      <td>39.206707</td>
      <td>38.87439</td>
      <td>38.914024</td>
      <td>38.753049</td>
      <td>38.712195</td>
      <td>38.596951</td>
      <td>38.492683</td>
      <td>38.453659</td>
      <td>38.418293</td>
      <td>38.418293</td>
    </tr>
  </tbody>
</table>
</div>




```python
# The country_averages df has a two-level index (country_year). We need to stack our gini dataframe to match
```


```python
gini_df = pd.DataFrame(gini_df.stack(), columns=["GINI coefficient"])

gini_df.reset_index(inplace=True)

gini_df.index = gini_df['Country Name'] + "_" + gini_df['level_1']

gini_df.drop(['Country Name', 'level_1'], axis = 1, inplace= True)

gini_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GINI coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Aruba_1980</th>
      <td>40.364024</td>
    </tr>
    <tr>
      <th>Aruba_1981</th>
      <td>40.364024</td>
    </tr>
    <tr>
      <th>Aruba_1982</th>
      <td>40.366463</td>
    </tr>
    <tr>
      <th>Aruba_1983</th>
      <td>40.370122</td>
    </tr>
    <tr>
      <th>Aruba_1984</th>
      <td>40.366463</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check our country_averages df has no nulls
assert country_averages.isnull().sum().sum() == 0
```


```python
# Merge in GINI information

job_and_indicator_df = pd.merge(job_and_gdp_df, gini_df, how = 'left', left_index = True, right_index= True)
```


```python
# Preview new dataframe

job_and_indicator_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Wave</th>
      <th>Year survey</th>
      <th>Country/region</th>
      <th>Important in a job: good pay</th>
      <th>Important in a job: not too much pressure</th>
      <th>Important in a job: good job security</th>
      <th>Important in a job: a respected job</th>
      <th>Important in a job: good hours</th>
      <th>Important in a job: an opportunity to use initiative</th>
      <th>Important in a job: generous holidays</th>
      <th>Important in a job: that you can achieve something</th>
      <th>Important in a job: a responsible job</th>
      <th>Important in a job: a job that is interesting</th>
      <th>Important in a job: a job that meets one´s abilities</th>
      <th>number_of_jobs_qs_answered</th>
      <th>country_year</th>
      <th>country_name</th>
      <th>total_mentions</th>
      <th>normalized Important in a job: good pay</th>
      <th>normalized Important in a job: not too much pressure</th>
      <th>normalized Important in a job: good job security</th>
      <th>normalized Important in a job: a respected job</th>
      <th>normalized Important in a job: good hours</th>
      <th>normalized Important in a job: an opportunity to use initiative</th>
      <th>normalized Important in a job: generous holidays</th>
      <th>normalized Important in a job: that you can achieve something</th>
      <th>normalized Important in a job: a responsible job</th>
      <th>normalized Important in a job: a job that is interesting</th>
      <th>normalized Important in a job: a job that meets one´s abilities</th>
      <th>normalized total_mentions</th>
      <th>normalized achieving_responsible_initiative</th>
      <th>normalized hours_holidays_pressure</th>
      <th>normalized security_pay</th>
      <th>kmeans_labels</th>
      <th>agglom_labels</th>
      <th>dbscan_labels</th>
      <th>kmeans_value_cluster</th>
      <th>GDP per capita, PPP</th>
      <th>GINI coefficient</th>
    </tr>
    <tr>
      <th>country_year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Albania_1998</th>
      <td>3.0</td>
      <td>1998</td>
      <td>8.0</td>
      <td>0.994995</td>
      <td>0.372372</td>
      <td>0.853854</td>
      <td>0.417417</td>
      <td>0.610611</td>
      <td>0.426426</td>
      <td>0.533534</td>
      <td>0.521522</td>
      <td>0.173173</td>
      <td>0.458458</td>
      <td>0.888889</td>
      <td>11.0</td>
      <td>Albania_1998</td>
      <td>Albania</td>
      <td>6.251251</td>
      <td>15.916733</td>
      <td>5.956765</td>
      <td>13.658927</td>
      <td>6.677342</td>
      <td>9.767814</td>
      <td>6.821457</td>
      <td>8.534828</td>
      <td>8.342674</td>
      <td>2.770216</td>
      <td>7.333867</td>
      <td>14.219376</td>
      <td>100.0</td>
      <td>17.934347</td>
      <td>24.259408</td>
      <td>29.575661</td>
      <td>2</td>
      <td>0</td>
      <td>-1</td>
      <td>lifestyle</td>
      <td>3209.615372</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>Albania_2002</th>
      <td>4.0</td>
      <td>2002</td>
      <td>8.0</td>
      <td>0.952000</td>
      <td>0.455000</td>
      <td>0.812000</td>
      <td>0.645000</td>
      <td>0.550000</td>
      <td>0.397000</td>
      <td>0.483000</td>
      <td>0.513000</td>
      <td>0.254000</td>
      <td>0.411000</td>
      <td>0.294000</td>
      <td>11.0</td>
      <td>Albania_2002</td>
      <td>Albania</td>
      <td>5.766000</td>
      <td>16.510579</td>
      <td>7.891086</td>
      <td>14.082553</td>
      <td>11.186264</td>
      <td>9.538675</td>
      <td>6.885189</td>
      <td>8.376691</td>
      <td>8.896982</td>
      <td>4.405134</td>
      <td>7.127992</td>
      <td>5.098855</td>
      <td>100.0</td>
      <td>20.187305</td>
      <td>25.806452</td>
      <td>30.593132</td>
      <td>2</td>
      <td>0</td>
      <td>-1</td>
      <td>lifestyle</td>
      <td>4754.653697</td>
      <td>31.7</td>
    </tr>
    <tr>
      <th>Algeria_2002</th>
      <td>4.0</td>
      <td>2002</td>
      <td>12.0</td>
      <td>0.897816</td>
      <td>0.595164</td>
      <td>0.863495</td>
      <td>0.719969</td>
      <td>0.511700</td>
      <td>0.435257</td>
      <td>0.205148</td>
      <td>0.602964</td>
      <td>0.464119</td>
      <td>0.565523</td>
      <td>0.701248</td>
      <td>11.0</td>
      <td>Algeria_2002</td>
      <td>Algeria</td>
      <td>6.562402</td>
      <td>13.681208</td>
      <td>9.069298</td>
      <td>13.158208</td>
      <td>10.971116</td>
      <td>7.797456</td>
      <td>6.632592</td>
      <td>3.126114</td>
      <td>9.188161</td>
      <td>7.072388</td>
      <td>8.617616</td>
      <td>10.685843</td>
      <td>100.0</td>
      <td>22.893142</td>
      <td>19.992868</td>
      <td>26.839415</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>lifestyle</td>
      <td>8911.680680</td>
      <td>35.3</td>
    </tr>
    <tr>
      <th>Armenia_1997</th>
      <td>3.0</td>
      <td>1997</td>
      <td>51.0</td>
      <td>0.935000</td>
      <td>0.530000</td>
      <td>0.773000</td>
      <td>0.698000</td>
      <td>0.487000</td>
      <td>0.456500</td>
      <td>0.306500</td>
      <td>0.655500</td>
      <td>0.427500</td>
      <td>0.773500</td>
      <td>0.662000</td>
      <td>11.0</td>
      <td>Armenia_1997</td>
      <td>Armenia</td>
      <td>6.704500</td>
      <td>13.945857</td>
      <td>7.905138</td>
      <td>11.529570</td>
      <td>10.410918</td>
      <td>7.263778</td>
      <td>6.808860</td>
      <td>4.571556</td>
      <td>9.777015</td>
      <td>6.376314</td>
      <td>11.537027</td>
      <td>9.873965</td>
      <td>100.0</td>
      <td>22.962190</td>
      <td>19.740473</td>
      <td>25.475427</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>lifestyle</td>
      <td>1843.378545</td>
      <td>36.2</td>
    </tr>
    <tr>
      <th>Azerbaijan_1997</th>
      <td>3.0</td>
      <td>1997</td>
      <td>31.0</td>
      <td>0.952048</td>
      <td>0.557443</td>
      <td>0.742757</td>
      <td>0.648851</td>
      <td>0.546953</td>
      <td>0.351648</td>
      <td>0.383616</td>
      <td>0.490010</td>
      <td>0.446553</td>
      <td>0.812687</td>
      <td>0.734266</td>
      <td>11.0</td>
      <td>Azerbaijan_1997</td>
      <td>Azerbaijan</td>
      <td>6.666833</td>
      <td>14.280363</td>
      <td>8.361430</td>
      <td>11.141080</td>
      <td>9.732524</td>
      <td>8.204091</td>
      <td>5.274594</td>
      <td>5.754102</td>
      <td>7.349966</td>
      <td>6.698134</td>
      <td>12.190005</td>
      <td>11.013711</td>
      <td>100.0</td>
      <td>19.322694</td>
      <td>22.319622</td>
      <td>25.421443</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>lifestyle</td>
      <td>2634.158631</td>
      <td>34.7</td>
    </tr>
  </tbody>
</table>
</div>



**Assess introduction of null values**


```python
job_and_indicator_df.isnull().sum().sum()
```




    0



- We had introduced 22 nulls, out of 93 values. We can fix many of these by matching our country names to country names in the World Bank database (e.g., Dominican Republic vs. Dominican Rep.). 
- Post-processing, 5 nulls remained. These are due to a handful of countries that do not report GINI scores (New Zealand, Puerto Rico, Saudi Arabia, Singapore)
- These remaining nulls were imputed with the average GINI coeff across all countries, leading to no additional nulls being inserted into our data

#### Merge in indicator data - primary school completion


```python
# Read in dataframe (from https://data.worldbank.org/, accessed July 2018)
primary_completion_df = pd.read_csv('./indicator_data/primary_completion.csv')
```


```python
# Preview data
primary_completion_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country Name</th>
      <th>1970</th>
      <th>1971</th>
      <th>1972</th>
      <th>1973</th>
      <th>1974</th>
      <th>1975</th>
      <th>1976</th>
      <th>1977</th>
      <th>1978</th>
      <th>1979</th>
      <th>1980</th>
      <th>1981</th>
      <th>1982</th>
      <th>1983</th>
      <th>1984</th>
      <th>1985</th>
      <th>1986</th>
      <th>1987</th>
      <th>1988</th>
      <th>1989</th>
      <th>1990</th>
      <th>1991</th>
      <th>1992</th>
      <th>1993</th>
      <th>1994</th>
      <th>1995</th>
      <th>1996</th>
      <th>1997</th>
      <th>1998</th>
      <th>1999</th>
      <th>2000</th>
      <th>2001</th>
      <th>2002</th>
      <th>2003</th>
      <th>2004</th>
      <th>2005</th>
      <th>2006</th>
      <th>2007</th>
      <th>2008</th>
      <th>2009</th>
      <th>2010</th>
      <th>2011</th>
      <th>2012</th>
      <th>2013</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
      <th>2017</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aruba</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>101.619118</td>
      <td>97.142860</td>
      <td>94.404068</td>
      <td>94.755241</td>
      <td>90.215919</td>
      <td>90.559898</td>
      <td>88.197968</td>
      <td>93.186119</td>
      <td>95.588242</td>
      <td>95.133034</td>
      <td>96.242577</td>
      <td>94.786102</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>96.570641</td>
      <td>101.180557</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>16.794571</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.43013</td>
      <td>19.107969</td>
      <td>NaN</td>
      <td>26.61916</td>
      <td>33.06739</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.200041</td>
      <td>18.909081</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29.069651</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Angola</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.09705</td>
      <td>36.053768</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.5553</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>39.270191</td>
      <td>38.839180</td>
      <td>39.806419</td>
      <td>46.086109</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albania</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>92.024391</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>97.073212</td>
      <td>96.880661</td>
      <td>NaN</td>
      <td>94.902428</td>
      <td>93.870880</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>96.629097</td>
      <td>92.830223</td>
      <td>91.512283</td>
      <td>89.237343</td>
      <td>93.277008</td>
      <td>98.911697</td>
      <td>90.974411</td>
      <td>105.002121</td>
      <td>104.468193</td>
      <td>105.604103</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Andorra</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Impute nulls
primary_completion_df.index = primary_completion_df['Country Name']
primary_completion_df.drop('Country Name', axis = 1, inplace=True)
primary_completion_df.fillna(method = 'ffill', axis=1, inplace=True)         # first try a forward fill (i.e., base on the last available value)
primary_completion_df.fillna(method = 'backfill', axis=1, inplace= True)     # next try a backward fill (i.e., base on the next available value)

primary_completion_df.fillna(value = 100, inplace=True)           # three countries have no data available for any years:
# the United States, Australia and Bosnia. Since primary education is compulsory and free in these countries,
# we impute a value of close to 100%
```


```python
primary_completion_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1970</th>
      <th>1971</th>
      <th>1972</th>
      <th>1973</th>
      <th>1974</th>
      <th>1975</th>
      <th>1976</th>
      <th>1977</th>
      <th>1978</th>
      <th>1979</th>
      <th>1980</th>
      <th>1981</th>
      <th>1982</th>
      <th>1983</th>
      <th>1984</th>
      <th>1985</th>
      <th>1986</th>
      <th>1987</th>
      <th>1988</th>
      <th>1989</th>
      <th>1990</th>
      <th>1991</th>
      <th>1992</th>
      <th>1993</th>
      <th>1994</th>
      <th>1995</th>
      <th>1996</th>
      <th>1997</th>
      <th>1998</th>
      <th>1999</th>
      <th>2000</th>
      <th>2001</th>
      <th>2002</th>
      <th>2003</th>
      <th>2004</th>
      <th>2005</th>
      <th>2006</th>
      <th>2007</th>
      <th>2008</th>
      <th>2009</th>
      <th>2010</th>
      <th>2011</th>
      <th>2012</th>
      <th>2013</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
      <th>2017</th>
    </tr>
    <tr>
      <th>Country Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Aruba</th>
      <td>101.619118</td>
      <td>101.619118</td>
      <td>101.619118</td>
      <td>101.619118</td>
      <td>101.619118</td>
      <td>101.619118</td>
      <td>101.619118</td>
      <td>101.619118</td>
      <td>101.619118</td>
      <td>101.619118</td>
      <td>101.619118</td>
      <td>101.619118</td>
      <td>101.619118</td>
      <td>101.619118</td>
      <td>101.619118</td>
      <td>101.619118</td>
      <td>101.619118</td>
      <td>101.619118</td>
      <td>101.619118</td>
      <td>101.619118</td>
      <td>101.619118</td>
      <td>101.619118</td>
      <td>101.619118</td>
      <td>101.619118</td>
      <td>101.619118</td>
      <td>101.619118</td>
      <td>101.619118</td>
      <td>101.619118</td>
      <td>101.619118</td>
      <td>101.619118</td>
      <td>97.142860</td>
      <td>94.404068</td>
      <td>94.755241</td>
      <td>90.215919</td>
      <td>90.559898</td>
      <td>88.197968</td>
      <td>93.186119</td>
      <td>95.588242</td>
      <td>95.133034</td>
      <td>96.242577</td>
      <td>94.786102</td>
      <td>94.786102</td>
      <td>94.786102</td>
      <td>96.570641</td>
      <td>101.180557</td>
      <td>101.180557</td>
      <td>101.180557</td>
      <td>101.180557</td>
    </tr>
    <tr>
      <th>Afghanistan</th>
      <td>16.794571</td>
      <td>16.794571</td>
      <td>16.794571</td>
      <td>16.794571</td>
      <td>16.794571</td>
      <td>16.794571</td>
      <td>16.794571</td>
      <td>17.430130</td>
      <td>19.107969</td>
      <td>19.107969</td>
      <td>26.619160</td>
      <td>33.067390</td>
      <td>33.067390</td>
      <td>33.067390</td>
      <td>18.200041</td>
      <td>18.909081</td>
      <td>18.909081</td>
      <td>18.909081</td>
      <td>18.909081</td>
      <td>18.909081</td>
      <td>18.909081</td>
      <td>18.909081</td>
      <td>18.909081</td>
      <td>29.069651</td>
      <td>29.069651</td>
      <td>29.069651</td>
      <td>29.069651</td>
      <td>29.069651</td>
      <td>29.069651</td>
      <td>29.069651</td>
      <td>29.069651</td>
      <td>29.069651</td>
      <td>29.069651</td>
      <td>29.069651</td>
      <td>29.069651</td>
      <td>29.069651</td>
      <td>29.069651</td>
      <td>29.069651</td>
      <td>29.069651</td>
      <td>29.069651</td>
      <td>29.069651</td>
      <td>29.069651</td>
      <td>29.069651</td>
      <td>29.069651</td>
      <td>29.069651</td>
      <td>29.069651</td>
      <td>29.069651</td>
      <td>29.069651</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>41.097050</td>
      <td>41.097050</td>
      <td>41.097050</td>
      <td>41.097050</td>
      <td>41.097050</td>
      <td>41.097050</td>
      <td>41.097050</td>
      <td>41.097050</td>
      <td>41.097050</td>
      <td>41.097050</td>
      <td>41.097050</td>
      <td>41.097050</td>
      <td>36.053768</td>
      <td>36.053768</td>
      <td>36.053768</td>
      <td>36.053768</td>
      <td>36.053768</td>
      <td>36.053768</td>
      <td>36.053768</td>
      <td>36.053768</td>
      <td>36.053768</td>
      <td>28.555300</td>
      <td>28.555300</td>
      <td>28.555300</td>
      <td>28.555300</td>
      <td>28.555300</td>
      <td>28.555300</td>
      <td>28.555300</td>
      <td>28.555300</td>
      <td>28.555300</td>
      <td>28.555300</td>
      <td>28.555300</td>
      <td>28.555300</td>
      <td>28.555300</td>
      <td>28.555300</td>
      <td>28.555300</td>
      <td>28.555300</td>
      <td>28.555300</td>
      <td>39.270191</td>
      <td>38.839180</td>
      <td>39.806419</td>
      <td>46.086109</td>
      <td>46.086109</td>
      <td>46.086109</td>
      <td>46.086109</td>
      <td>46.086109</td>
      <td>46.086109</td>
      <td>46.086109</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>92.024391</td>
      <td>92.024391</td>
      <td>92.024391</td>
      <td>92.024391</td>
      <td>92.024391</td>
      <td>92.024391</td>
      <td>92.024391</td>
      <td>92.024391</td>
      <td>92.024391</td>
      <td>92.024391</td>
      <td>92.024391</td>
      <td>92.024391</td>
      <td>92.024391</td>
      <td>92.024391</td>
      <td>92.024391</td>
      <td>92.024391</td>
      <td>92.024391</td>
      <td>92.024391</td>
      <td>92.024391</td>
      <td>92.024391</td>
      <td>92.024391</td>
      <td>92.024391</td>
      <td>92.024391</td>
      <td>92.024391</td>
      <td>92.024391</td>
      <td>92.024391</td>
      <td>92.024391</td>
      <td>92.024391</td>
      <td>92.024391</td>
      <td>92.024391</td>
      <td>97.073212</td>
      <td>96.880661</td>
      <td>96.880661</td>
      <td>94.902428</td>
      <td>93.870880</td>
      <td>93.870880</td>
      <td>93.870880</td>
      <td>96.629097</td>
      <td>92.830223</td>
      <td>91.512283</td>
      <td>89.237343</td>
      <td>93.277008</td>
      <td>98.911697</td>
      <td>90.974411</td>
      <td>105.002121</td>
      <td>104.468193</td>
      <td>105.604103</td>
      <td>105.604103</td>
    </tr>
    <tr>
      <th>Andorra</th>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# The country_averages df has a two-level index (country_year). We need to stack our education dataframe to match
```


```python
primary_completion_df = pd.DataFrame(primary_completion_df.stack(), columns=["Primary completion rate"])

primary_completion_df.reset_index(inplace=True)

primary_completion_df.index = primary_completion_df['Country Name'] + "_" + primary_completion_df['level_1']

primary_completion_df.drop(['Country Name', 'level_1'], axis = 1, inplace= True)

primary_completion_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Primary completion rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Aruba_1970</th>
      <td>101.619118</td>
    </tr>
    <tr>
      <th>Aruba_1971</th>
      <td>101.619118</td>
    </tr>
    <tr>
      <th>Aruba_1972</th>
      <td>101.619118</td>
    </tr>
    <tr>
      <th>Aruba_1973</th>
      <td>101.619118</td>
    </tr>
    <tr>
      <th>Aruba_1974</th>
      <td>101.619118</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check our country_averages df has no nulls
assert job_and_indicator_df.isnull().sum().sum() == 0
```


```python
# Merge in GINI information

job_and_indicator_df = pd.merge(job_and_indicator_df, primary_completion_df, how = 'left', left_index = True, right_index= True)
```


```python
# Preview new dataframe

job_and_indicator_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Wave</th>
      <th>Year survey</th>
      <th>Country/region</th>
      <th>Important in a job: good pay</th>
      <th>Important in a job: not too much pressure</th>
      <th>Important in a job: good job security</th>
      <th>Important in a job: a respected job</th>
      <th>Important in a job: good hours</th>
      <th>Important in a job: an opportunity to use initiative</th>
      <th>Important in a job: generous holidays</th>
      <th>Important in a job: that you can achieve something</th>
      <th>Important in a job: a responsible job</th>
      <th>Important in a job: a job that is interesting</th>
      <th>Important in a job: a job that meets one´s abilities</th>
      <th>number_of_jobs_qs_answered</th>
      <th>country_year</th>
      <th>country_name</th>
      <th>total_mentions</th>
      <th>normalized Important in a job: good pay</th>
      <th>normalized Important in a job: not too much pressure</th>
      <th>normalized Important in a job: good job security</th>
      <th>normalized Important in a job: a respected job</th>
      <th>normalized Important in a job: good hours</th>
      <th>normalized Important in a job: an opportunity to use initiative</th>
      <th>normalized Important in a job: generous holidays</th>
      <th>normalized Important in a job: that you can achieve something</th>
      <th>normalized Important in a job: a responsible job</th>
      <th>normalized Important in a job: a job that is interesting</th>
      <th>normalized Important in a job: a job that meets one´s abilities</th>
      <th>normalized total_mentions</th>
      <th>normalized achieving_responsible_initiative</th>
      <th>normalized hours_holidays_pressure</th>
      <th>normalized security_pay</th>
      <th>kmeans_labels</th>
      <th>agglom_labels</th>
      <th>dbscan_labels</th>
      <th>kmeans_value_cluster</th>
      <th>GDP per capita, PPP</th>
      <th>GINI coefficient</th>
      <th>Primary completion rate</th>
    </tr>
    <tr>
      <th>country_year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Albania_1998</th>
      <td>3.0</td>
      <td>1998</td>
      <td>8.0</td>
      <td>0.994995</td>
      <td>0.372372</td>
      <td>0.853854</td>
      <td>0.417417</td>
      <td>0.610611</td>
      <td>0.426426</td>
      <td>0.533534</td>
      <td>0.521522</td>
      <td>0.173173</td>
      <td>0.458458</td>
      <td>0.888889</td>
      <td>11.0</td>
      <td>Albania_1998</td>
      <td>Albania</td>
      <td>6.251251</td>
      <td>15.916733</td>
      <td>5.956765</td>
      <td>13.658927</td>
      <td>6.677342</td>
      <td>9.767814</td>
      <td>6.821457</td>
      <td>8.534828</td>
      <td>8.342674</td>
      <td>2.770216</td>
      <td>7.333867</td>
      <td>14.219376</td>
      <td>100.0</td>
      <td>17.934347</td>
      <td>24.259408</td>
      <td>29.575661</td>
      <td>2</td>
      <td>0</td>
      <td>-1</td>
      <td>lifestyle</td>
      <td>3209.615372</td>
      <td>27.0</td>
      <td>92.024391</td>
    </tr>
    <tr>
      <th>Albania_2002</th>
      <td>4.0</td>
      <td>2002</td>
      <td>8.0</td>
      <td>0.952000</td>
      <td>0.455000</td>
      <td>0.812000</td>
      <td>0.645000</td>
      <td>0.550000</td>
      <td>0.397000</td>
      <td>0.483000</td>
      <td>0.513000</td>
      <td>0.254000</td>
      <td>0.411000</td>
      <td>0.294000</td>
      <td>11.0</td>
      <td>Albania_2002</td>
      <td>Albania</td>
      <td>5.766000</td>
      <td>16.510579</td>
      <td>7.891086</td>
      <td>14.082553</td>
      <td>11.186264</td>
      <td>9.538675</td>
      <td>6.885189</td>
      <td>8.376691</td>
      <td>8.896982</td>
      <td>4.405134</td>
      <td>7.127992</td>
      <td>5.098855</td>
      <td>100.0</td>
      <td>20.187305</td>
      <td>25.806452</td>
      <td>30.593132</td>
      <td>2</td>
      <td>0</td>
      <td>-1</td>
      <td>lifestyle</td>
      <td>4754.653697</td>
      <td>31.7</td>
      <td>96.880661</td>
    </tr>
    <tr>
      <th>Algeria_2002</th>
      <td>4.0</td>
      <td>2002</td>
      <td>12.0</td>
      <td>0.897816</td>
      <td>0.595164</td>
      <td>0.863495</td>
      <td>0.719969</td>
      <td>0.511700</td>
      <td>0.435257</td>
      <td>0.205148</td>
      <td>0.602964</td>
      <td>0.464119</td>
      <td>0.565523</td>
      <td>0.701248</td>
      <td>11.0</td>
      <td>Algeria_2002</td>
      <td>Algeria</td>
      <td>6.562402</td>
      <td>13.681208</td>
      <td>9.069298</td>
      <td>13.158208</td>
      <td>10.971116</td>
      <td>7.797456</td>
      <td>6.632592</td>
      <td>3.126114</td>
      <td>9.188161</td>
      <td>7.072388</td>
      <td>8.617616</td>
      <td>10.685843</td>
      <td>100.0</td>
      <td>22.893142</td>
      <td>19.992868</td>
      <td>26.839415</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>lifestyle</td>
      <td>8911.680680</td>
      <td>35.3</td>
      <td>87.843491</td>
    </tr>
    <tr>
      <th>Armenia_1997</th>
      <td>3.0</td>
      <td>1997</td>
      <td>51.0</td>
      <td>0.935000</td>
      <td>0.530000</td>
      <td>0.773000</td>
      <td>0.698000</td>
      <td>0.487000</td>
      <td>0.456500</td>
      <td>0.306500</td>
      <td>0.655500</td>
      <td>0.427500</td>
      <td>0.773500</td>
      <td>0.662000</td>
      <td>11.0</td>
      <td>Armenia_1997</td>
      <td>Armenia</td>
      <td>6.704500</td>
      <td>13.945857</td>
      <td>7.905138</td>
      <td>11.529570</td>
      <td>10.410918</td>
      <td>7.263778</td>
      <td>6.808860</td>
      <td>4.571556</td>
      <td>9.777015</td>
      <td>6.376314</td>
      <td>11.537027</td>
      <td>9.873965</td>
      <td>100.0</td>
      <td>22.962190</td>
      <td>19.740473</td>
      <td>25.475427</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>lifestyle</td>
      <td>1843.378545</td>
      <td>36.2</td>
      <td>93.895500</td>
    </tr>
    <tr>
      <th>Azerbaijan_1997</th>
      <td>3.0</td>
      <td>1997</td>
      <td>31.0</td>
      <td>0.952048</td>
      <td>0.557443</td>
      <td>0.742757</td>
      <td>0.648851</td>
      <td>0.546953</td>
      <td>0.351648</td>
      <td>0.383616</td>
      <td>0.490010</td>
      <td>0.446553</td>
      <td>0.812687</td>
      <td>0.734266</td>
      <td>11.0</td>
      <td>Azerbaijan_1997</td>
      <td>Azerbaijan</td>
      <td>6.666833</td>
      <td>14.280363</td>
      <td>8.361430</td>
      <td>11.141080</td>
      <td>9.732524</td>
      <td>8.204091</td>
      <td>5.274594</td>
      <td>5.754102</td>
      <td>7.349966</td>
      <td>6.698134</td>
      <td>12.190005</td>
      <td>11.013711</td>
      <td>100.0</td>
      <td>19.322694</td>
      <td>22.319622</td>
      <td>25.421443</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>lifestyle</td>
      <td>2634.158631</td>
      <td>34.7</td>
      <td>99.470108</td>
    </tr>
  </tbody>
</table>
</div>



**Assess introduction of null values**


```python
job_and_indicator_df.isnull().sum().sum()
```




    0



- We had introduced 53 nulls, out of 93 values. We can fix many of these by matching our country names to country names in the World Bank database (e.g., Dominican Republic vs. Dominican Rep.). 
- Post-processing, 43 nulls remained
- Using back and forward fill to impute missing values, we can reduce the number of nulls to 5
- These are due to a handful of countries that do not report primary school completion (Australia, Bosnia, United States)
- Primary education is mandatory and free for children in all these countries, so we can impute a completion rate of close to 100

#### Merge in indicator data - democracy index
Overall polity score from the Polity IV dataset, calculated by subtracting an autocracy score from a democracy score. It is a summary measure of a country's democratic and free nature. -10 is the lowest value, 10 the highest.


```python
# Read in dataframe (from https://www.gapminder.org/data/ ('Democracy Index'), accessed July 2018)
democracy_df = pd.read_csv("./indicator_data/Democracy_index_gapminder.csv")
```


```python
# Preview data
democracy_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Democracy index</th>
      <th>1800</th>
      <th>1801</th>
      <th>1802</th>
      <th>1803</th>
      <th>1804</th>
      <th>1805</th>
      <th>1806</th>
      <th>1807</th>
      <th>1808</th>
      <th>1809</th>
      <th>1810</th>
      <th>1811</th>
      <th>1812</th>
      <th>1813</th>
      <th>1814</th>
      <th>1815</th>
      <th>1816</th>
      <th>1817</th>
      <th>1818</th>
      <th>1819</th>
      <th>1820</th>
      <th>1821</th>
      <th>1822</th>
      <th>1823</th>
      <th>1824</th>
      <th>1825</th>
      <th>1826</th>
      <th>1827</th>
      <th>1828</th>
      <th>1829</th>
      <th>1830</th>
      <th>1831</th>
      <th>1832</th>
      <th>1833</th>
      <th>1834</th>
      <th>1835</th>
      <th>1836</th>
      <th>1837</th>
      <th>1838</th>
      <th>1839</th>
      <th>1840</th>
      <th>1841</th>
      <th>1842</th>
      <th>1843</th>
      <th>1844</th>
      <th>1845</th>
      <th>1846</th>
      <th>1847</th>
      <th>1848</th>
      <th>1849</th>
      <th>1850</th>
      <th>1851</th>
      <th>1852</th>
      <th>1853</th>
      <th>1854</th>
      <th>1855</th>
      <th>1856</th>
      <th>1857</th>
      <th>1858</th>
      <th>1859</th>
      <th>1860</th>
      <th>1861</th>
      <th>1862</th>
      <th>1863</th>
      <th>1864</th>
      <th>1865</th>
      <th>1866</th>
      <th>1867</th>
      <th>1868</th>
      <th>1869</th>
      <th>1870</th>
      <th>1871</th>
      <th>1872</th>
      <th>1873</th>
      <th>1874</th>
      <th>1875</th>
      <th>1876</th>
      <th>1877</th>
      <th>1878</th>
      <th>1879</th>
      <th>1880</th>
      <th>1881</th>
      <th>1882</th>
      <th>1883</th>
      <th>1884</th>
      <th>1885</th>
      <th>1886</th>
      <th>1887</th>
      <th>1888</th>
      <th>1889</th>
      <th>1890</th>
      <th>1891</th>
      <th>1892</th>
      <th>1893</th>
      <th>1894</th>
      <th>1895</th>
      <th>1896</th>
      <th>1897</th>
      <th>1898</th>
      <th>1899</th>
      <th>1900</th>
      <th>1901</th>
      <th>1902</th>
      <th>1903</th>
      <th>1904</th>
      <th>1905</th>
      <th>1906</th>
      <th>1907</th>
      <th>1908</th>
      <th>1909</th>
      <th>1910</th>
      <th>1911</th>
      <th>1912</th>
      <th>1913</th>
      <th>1914</th>
      <th>1915</th>
      <th>1916</th>
      <th>1917</th>
      <th>1918</th>
      <th>1919</th>
      <th>1920</th>
      <th>1921</th>
      <th>1922</th>
      <th>1923</th>
      <th>1924</th>
      <th>1925</th>
      <th>1926</th>
      <th>1927</th>
      <th>1928</th>
      <th>1929</th>
      <th>1930</th>
      <th>1931</th>
      <th>1932</th>
      <th>1933</th>
      <th>1934</th>
      <th>1935</th>
      <th>1936</th>
      <th>1937</th>
      <th>1938</th>
      <th>1939</th>
      <th>1940</th>
      <th>1941</th>
      <th>1942</th>
      <th>1943</th>
      <th>1944</th>
      <th>1945</th>
      <th>1946</th>
      <th>1947</th>
      <th>1948</th>
      <th>1949</th>
      <th>1950</th>
      <th>1951</th>
      <th>1952</th>
      <th>1953</th>
      <th>1954</th>
      <th>1955</th>
      <th>1956</th>
      <th>1957</th>
      <th>1958</th>
      <th>1959</th>
      <th>1960</th>
      <th>1961</th>
      <th>1962</th>
      <th>1963</th>
      <th>1964</th>
      <th>1965</th>
      <th>1966</th>
      <th>1967</th>
      <th>1968</th>
      <th>1969</th>
      <th>1970</th>
      <th>1971</th>
      <th>1972</th>
      <th>1973</th>
      <th>1974</th>
      <th>1975</th>
      <th>1976</th>
      <th>1977</th>
      <th>1978</th>
      <th>1979</th>
      <th>1980</th>
      <th>1981</th>
      <th>1982</th>
      <th>1983</th>
      <th>1984</th>
      <th>1985</th>
      <th>1986</th>
      <th>1987</th>
      <th>1988</th>
      <th>1989</th>
      <th>1990</th>
      <th>1991</th>
      <th>1992</th>
      <th>1993</th>
      <th>1994</th>
      <th>1995</th>
      <th>1996</th>
      <th>1997</th>
      <th>1998</th>
      <th>1999</th>
      <th>2000</th>
      <th>2001</th>
      <th>2002</th>
      <th>2003</th>
      <th>2004</th>
      <th>2005</th>
      <th>2006</th>
      <th>2007</th>
      <th>2008</th>
      <th>2009</th>
      <th>2010</th>
      <th>2011</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Abkhazia</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-6.0</td>
      <td>-8.0</td>
      <td>-8.0</td>
      <td>-8.0</td>
      <td>-8.0</td>
      <td>-8.0</td>
      <td>-8.0</td>
      <td>-8.0</td>
      <td>-8.0</td>
      <td>-8.0</td>
      <td>-8.0</td>
      <td>-10.0</td>
      <td>-10.0</td>
      <td>-10.0</td>
      <td>-10.0</td>
      <td>-10.0</td>
      <td>-10.0</td>
      <td>-10.0</td>
      <td>-10.0</td>
      <td>-10.0</td>
      <td>-10.0</td>
      <td>-10.0</td>
      <td>-10.0</td>
      <td>-10.0</td>
      <td>-10.0</td>
      <td>-10.0</td>
      <td>-10.0</td>
      <td>-10.0</td>
      <td>-10.0</td>
      <td>-10.0</td>
      <td>-7.0</td>
      <td>-7.0</td>
      <td>-7.0</td>
      <td>-7.0</td>
      <td>-7.0</td>
      <td>-7.0</td>
      <td>-7.0</td>
      <td>-7.0</td>
      <td>-7.0</td>
      <td>-7.0</td>
      <td>-7.0</td>
      <td>-7.0</td>
      <td>-7.0</td>
      <td>-7.0</td>
      <td>0.0</td>
      <td>-10.0</td>
      <td>-10.0</td>
      <td>-10.0</td>
      <td>-10.0</td>
      <td>-10.0</td>
      <td>-10.0</td>
      <td>-10.0</td>
      <td>-10.0</td>
      <td>-10.0</td>
      <td>-10.0</td>
      <td>-8.0</td>
      <td>-8.0</td>
      <td>-8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-7.0</td>
      <td>-7.0</td>
      <td>-7.0</td>
      <td>-7.0</td>
      <td>-7.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Akrotiri and Dhekelia</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albania</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-5.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Algeria</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-8.0</td>
      <td>-8.0</td>
      <td>-8.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-9.0</td>
      <td>-2.0</td>
      <td>-2.0</td>
      <td>-2.0</td>
      <td>-7.0</td>
      <td>-7.0</td>
      <td>-7.0</td>
      <td>-3.0</td>
      <td>-3.0</td>
      <td>-3.0</td>
      <td>-3.0</td>
      <td>-3.0</td>
      <td>-3.0</td>
      <td>-3.0</td>
      <td>-3.0</td>
      <td>-3.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Example country: UK
# democracy_df.index = democracy_df['Democracy index']
# UK = pd.DataFrame(democracy_df.loc['United Kingdom', :])
# UK.T
```


```python
# Impute nulls
democracy_df.rename(columns = {'Democracy index':'Country Name'}, inplace=True)   # rename our country labels to match prior dataframes
democracy_df.index = democracy_df['Country Name']
democracy_df.drop('Country Name', axis = 1, inplace=True)
democracy_df.fillna(method = 'ffill', axis=1, inplace=True)                     # first try a forward fill (i.e., base on the last available value)
democracy_df.fillna(method = 'backfill', axis=1, inplace= True)                 # next try a backward fill (i.e., base on the next available value)

democracy_df.fillna(value = democracy_df.mean(), inplace=True)
```


```python
# The country_averages df has a two-level index (country_year). We need to stack our democracy dataframe to match
```


```python
democracy_df = pd.DataFrame(democracy_df.stack(), columns=["Democracy index"])

democracy_df.reset_index(inplace=True)

democracy_df.index = democracy_df['Country Name'] + "_" + democracy_df['level_1']

democracy_df.drop(['Country Name', 'level_1'], axis = 1, inplace= True)

democracy_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Democracy index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Abkhazia_1800</th>
      <td>-2.020942</td>
    </tr>
    <tr>
      <th>Abkhazia_1801</th>
      <td>-2.020942</td>
    </tr>
    <tr>
      <th>Abkhazia_1802</th>
      <td>-2.020942</td>
    </tr>
    <tr>
      <th>Abkhazia_1803</th>
      <td>-2.020942</td>
    </tr>
    <tr>
      <th>Abkhazia_1804</th>
      <td>-2.020942</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check our country_averages df has no nulls
assert job_and_indicator_df.isnull().sum().sum() == 0
```


```python
# Merge in democracy information

job_and_indicator_df = pd.merge(job_and_indicator_df, democracy_df, how = 'left', left_index = True, right_index= True)
```


```python
# Preview new dataframe

job_and_indicator_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Wave</th>
      <th>Year survey</th>
      <th>Country/region</th>
      <th>Important in a job: good pay</th>
      <th>Important in a job: not too much pressure</th>
      <th>Important in a job: good job security</th>
      <th>Important in a job: a respected job</th>
      <th>Important in a job: good hours</th>
      <th>Important in a job: an opportunity to use initiative</th>
      <th>Important in a job: generous holidays</th>
      <th>Important in a job: that you can achieve something</th>
      <th>Important in a job: a responsible job</th>
      <th>Important in a job: a job that is interesting</th>
      <th>Important in a job: a job that meets one´s abilities</th>
      <th>number_of_jobs_qs_answered</th>
      <th>country_year</th>
      <th>country_name</th>
      <th>total_mentions</th>
      <th>normalized Important in a job: good pay</th>
      <th>normalized Important in a job: not too much pressure</th>
      <th>normalized Important in a job: good job security</th>
      <th>normalized Important in a job: a respected job</th>
      <th>normalized Important in a job: good hours</th>
      <th>normalized Important in a job: an opportunity to use initiative</th>
      <th>normalized Important in a job: generous holidays</th>
      <th>normalized Important in a job: that you can achieve something</th>
      <th>normalized Important in a job: a responsible job</th>
      <th>normalized Important in a job: a job that is interesting</th>
      <th>normalized Important in a job: a job that meets one´s abilities</th>
      <th>normalized total_mentions</th>
      <th>normalized achieving_responsible_initiative</th>
      <th>normalized hours_holidays_pressure</th>
      <th>normalized security_pay</th>
      <th>kmeans_labels</th>
      <th>agglom_labels</th>
      <th>dbscan_labels</th>
      <th>kmeans_value_cluster</th>
      <th>GDP per capita, PPP</th>
      <th>GINI coefficient</th>
      <th>Primary completion rate</th>
      <th>Democracy index</th>
    </tr>
    <tr>
      <th>country_year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Albania_1998</th>
      <td>3.0</td>
      <td>1998</td>
      <td>8.0</td>
      <td>0.994995</td>
      <td>0.372372</td>
      <td>0.853854</td>
      <td>0.417417</td>
      <td>0.610611</td>
      <td>0.426426</td>
      <td>0.533534</td>
      <td>0.521522</td>
      <td>0.173173</td>
      <td>0.458458</td>
      <td>0.888889</td>
      <td>11.0</td>
      <td>Albania_1998</td>
      <td>Albania</td>
      <td>6.251251</td>
      <td>15.916733</td>
      <td>5.956765</td>
      <td>13.658927</td>
      <td>6.677342</td>
      <td>9.767814</td>
      <td>6.821457</td>
      <td>8.534828</td>
      <td>8.342674</td>
      <td>2.770216</td>
      <td>7.333867</td>
      <td>14.219376</td>
      <td>100.0</td>
      <td>17.934347</td>
      <td>24.259408</td>
      <td>29.575661</td>
      <td>2</td>
      <td>0</td>
      <td>-1</td>
      <td>lifestyle</td>
      <td>3209.615372</td>
      <td>27.0</td>
      <td>92.024391</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Albania_2002</th>
      <td>4.0</td>
      <td>2002</td>
      <td>8.0</td>
      <td>0.952000</td>
      <td>0.455000</td>
      <td>0.812000</td>
      <td>0.645000</td>
      <td>0.550000</td>
      <td>0.397000</td>
      <td>0.483000</td>
      <td>0.513000</td>
      <td>0.254000</td>
      <td>0.411000</td>
      <td>0.294000</td>
      <td>11.0</td>
      <td>Albania_2002</td>
      <td>Albania</td>
      <td>5.766000</td>
      <td>16.510579</td>
      <td>7.891086</td>
      <td>14.082553</td>
      <td>11.186264</td>
      <td>9.538675</td>
      <td>6.885189</td>
      <td>8.376691</td>
      <td>8.896982</td>
      <td>4.405134</td>
      <td>7.127992</td>
      <td>5.098855</td>
      <td>100.0</td>
      <td>20.187305</td>
      <td>25.806452</td>
      <td>30.593132</td>
      <td>2</td>
      <td>0</td>
      <td>-1</td>
      <td>lifestyle</td>
      <td>4754.653697</td>
      <td>31.7</td>
      <td>96.880661</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>Algeria_2002</th>
      <td>4.0</td>
      <td>2002</td>
      <td>12.0</td>
      <td>0.897816</td>
      <td>0.595164</td>
      <td>0.863495</td>
      <td>0.719969</td>
      <td>0.511700</td>
      <td>0.435257</td>
      <td>0.205148</td>
      <td>0.602964</td>
      <td>0.464119</td>
      <td>0.565523</td>
      <td>0.701248</td>
      <td>11.0</td>
      <td>Algeria_2002</td>
      <td>Algeria</td>
      <td>6.562402</td>
      <td>13.681208</td>
      <td>9.069298</td>
      <td>13.158208</td>
      <td>10.971116</td>
      <td>7.797456</td>
      <td>6.632592</td>
      <td>3.126114</td>
      <td>9.188161</td>
      <td>7.072388</td>
      <td>8.617616</td>
      <td>10.685843</td>
      <td>100.0</td>
      <td>22.893142</td>
      <td>19.992868</td>
      <td>26.839415</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>lifestyle</td>
      <td>8911.680680</td>
      <td>35.3</td>
      <td>87.843491</td>
      <td>-3.0</td>
    </tr>
    <tr>
      <th>Armenia_1997</th>
      <td>3.0</td>
      <td>1997</td>
      <td>51.0</td>
      <td>0.935000</td>
      <td>0.530000</td>
      <td>0.773000</td>
      <td>0.698000</td>
      <td>0.487000</td>
      <td>0.456500</td>
      <td>0.306500</td>
      <td>0.655500</td>
      <td>0.427500</td>
      <td>0.773500</td>
      <td>0.662000</td>
      <td>11.0</td>
      <td>Armenia_1997</td>
      <td>Armenia</td>
      <td>6.704500</td>
      <td>13.945857</td>
      <td>7.905138</td>
      <td>11.529570</td>
      <td>10.410918</td>
      <td>7.263778</td>
      <td>6.808860</td>
      <td>4.571556</td>
      <td>9.777015</td>
      <td>6.376314</td>
      <td>11.537027</td>
      <td>9.873965</td>
      <td>100.0</td>
      <td>22.962190</td>
      <td>19.740473</td>
      <td>25.475427</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>lifestyle</td>
      <td>1843.378545</td>
      <td>36.2</td>
      <td>93.895500</td>
      <td>-6.0</td>
    </tr>
    <tr>
      <th>Azerbaijan_1997</th>
      <td>3.0</td>
      <td>1997</td>
      <td>31.0</td>
      <td>0.952048</td>
      <td>0.557443</td>
      <td>0.742757</td>
      <td>0.648851</td>
      <td>0.546953</td>
      <td>0.351648</td>
      <td>0.383616</td>
      <td>0.490010</td>
      <td>0.446553</td>
      <td>0.812687</td>
      <td>0.734266</td>
      <td>11.0</td>
      <td>Azerbaijan_1997</td>
      <td>Azerbaijan</td>
      <td>6.666833</td>
      <td>14.280363</td>
      <td>8.361430</td>
      <td>11.141080</td>
      <td>9.732524</td>
      <td>8.204091</td>
      <td>5.274594</td>
      <td>5.754102</td>
      <td>7.349966</td>
      <td>6.698134</td>
      <td>12.190005</td>
      <td>11.013711</td>
      <td>100.0</td>
      <td>19.322694</td>
      <td>22.319622</td>
      <td>25.421443</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>lifestyle</td>
      <td>2634.158631</td>
      <td>34.7</td>
      <td>99.470108</td>
      <td>-6.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
job_and_indicator_df.shape
```




    (93, 41)



**Assess introduction of null values**


```python
job_and_indicator_df.isnull().sum().sum()
```




    0




```python
# job_and_indicator_df.sort_values('Democracy index', na_position='first')
```

- We had introduced 10 nulls, out of 93 values. We can fix many of these by matching our country names to country names in the World Bank database (e.g., Dominican Republic vs. Dominican Rep.). 
- Post-processing, 5 nulls remained
- Using back and forward fill to impute missing values, we can reduce the number of nulls to 2
- There is no score for Puerto Rico, but we can impute this with the mean for the dataframe

#### Create baseline for model

We have identified three clusters of attitudes towards jobs - the lifestyle group, the achievement group and the balanced group. Our model will be a logistic regression model to assess whether we can predict which cluster a country is in at a given point in time, based on its GDP.

The simplest model we could create is to predict the majority class each time


```python
# Identifying the majority class
job_and_gdp_df['kmeans_labels'].value_counts()
```




    0    38
    2    32
    1    23
    Name: kmeans_labels, dtype: int64




```python
# Baseline accuracy (i.e., the accuracy of a model that just predicts the majority class)
print(f'Baseline accuracy = {max(job_and_gdp_df["kmeans_labels"].value_counts())/ job_and_gdp_df.shape[0]*100:.0f}%')
```

    Baseline accuracy = 41%


#### Normalize features to allow for feature importance selection


```python
indicator_features = ['GDP per capita, PPP', 'GINI coefficient', 'Primary completion rate', 'Democracy index']
```


```python
ss = StandardScaler()
```


```python
scaled_indicator_columns = pd.DataFrame(ss.fit_transform(job_and_indicator_df[indicator_features]),
                                       columns = ['Scaled GDP per capita, PPP', 'Scaled GINI coefficient', 'Scaled Primary completion rate', 'Scaled Democracy index'],
                                       index = job_and_indicator_df.index)
```


```python
job_and_indicator_df = pd.merge(job_and_indicator_df, scaled_indicator_columns, left_index=True, right_index=True)
```


```python
job_and_indicator_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Wave</th>
      <th>Year survey</th>
      <th>Country/region</th>
      <th>Important in a job: good pay</th>
      <th>Important in a job: not too much pressure</th>
      <th>Important in a job: good job security</th>
      <th>Important in a job: a respected job</th>
      <th>Important in a job: good hours</th>
      <th>Important in a job: an opportunity to use initiative</th>
      <th>Important in a job: generous holidays</th>
      <th>Important in a job: that you can achieve something</th>
      <th>Important in a job: a responsible job</th>
      <th>Important in a job: a job that is interesting</th>
      <th>Important in a job: a job that meets one´s abilities</th>
      <th>number_of_jobs_qs_answered</th>
      <th>country_year</th>
      <th>country_name</th>
      <th>total_mentions</th>
      <th>normalized Important in a job: good pay</th>
      <th>normalized Important in a job: not too much pressure</th>
      <th>normalized Important in a job: good job security</th>
      <th>normalized Important in a job: a respected job</th>
      <th>normalized Important in a job: good hours</th>
      <th>normalized Important in a job: an opportunity to use initiative</th>
      <th>normalized Important in a job: generous holidays</th>
      <th>normalized Important in a job: that you can achieve something</th>
      <th>normalized Important in a job: a responsible job</th>
      <th>normalized Important in a job: a job that is interesting</th>
      <th>normalized Important in a job: a job that meets one´s abilities</th>
      <th>normalized total_mentions</th>
      <th>normalized achieving_responsible_initiative</th>
      <th>normalized hours_holidays_pressure</th>
      <th>normalized security_pay</th>
      <th>kmeans_labels</th>
      <th>agglom_labels</th>
      <th>dbscan_labels</th>
      <th>kmeans_value_cluster</th>
      <th>GDP per capita, PPP</th>
      <th>GINI coefficient</th>
      <th>Primary completion rate</th>
      <th>Democracy index</th>
      <th>Scaled GDP per capita, PPP</th>
      <th>Scaled GINI coefficient</th>
      <th>Scaled Primary completion rate</th>
      <th>Scaled Democracy index</th>
    </tr>
    <tr>
      <th>country_year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Albania_1998</th>
      <td>3.0</td>
      <td>1998</td>
      <td>8.0</td>
      <td>0.994995</td>
      <td>0.372372</td>
      <td>0.853854</td>
      <td>0.417417</td>
      <td>0.610611</td>
      <td>0.426426</td>
      <td>0.533534</td>
      <td>0.521522</td>
      <td>0.173173</td>
      <td>0.458458</td>
      <td>0.888889</td>
      <td>11.0</td>
      <td>Albania_1998</td>
      <td>Albania</td>
      <td>6.251251</td>
      <td>15.916733</td>
      <td>5.956765</td>
      <td>13.658927</td>
      <td>6.677342</td>
      <td>9.767814</td>
      <td>6.821457</td>
      <td>8.534828</td>
      <td>8.342674</td>
      <td>2.770216</td>
      <td>7.333867</td>
      <td>14.219376</td>
      <td>100.0</td>
      <td>17.934347</td>
      <td>24.259408</td>
      <td>29.575661</td>
      <td>2</td>
      <td>0</td>
      <td>-1</td>
      <td>lifestyle</td>
      <td>3209.615372</td>
      <td>27.0</td>
      <td>92.024391</td>
      <td>5.0</td>
      <td>-0.747899</td>
      <td>-1.370790</td>
      <td>0.110121</td>
      <td>0.129798</td>
    </tr>
    <tr>
      <th>Albania_2002</th>
      <td>4.0</td>
      <td>2002</td>
      <td>8.0</td>
      <td>0.952000</td>
      <td>0.455000</td>
      <td>0.812000</td>
      <td>0.645000</td>
      <td>0.550000</td>
      <td>0.397000</td>
      <td>0.483000</td>
      <td>0.513000</td>
      <td>0.254000</td>
      <td>0.411000</td>
      <td>0.294000</td>
      <td>11.0</td>
      <td>Albania_2002</td>
      <td>Albania</td>
      <td>5.766000</td>
      <td>16.510579</td>
      <td>7.891086</td>
      <td>14.082553</td>
      <td>11.186264</td>
      <td>9.538675</td>
      <td>6.885189</td>
      <td>8.376691</td>
      <td>8.896982</td>
      <td>4.405134</td>
      <td>7.127992</td>
      <td>5.098855</td>
      <td>100.0</td>
      <td>20.187305</td>
      <td>25.806452</td>
      <td>30.593132</td>
      <td>2</td>
      <td>0</td>
      <td>-1</td>
      <td>lifestyle</td>
      <td>4754.653697</td>
      <td>31.7</td>
      <td>96.880661</td>
      <td>7.0</td>
      <td>-0.578766</td>
      <td>-0.813507</td>
      <td>0.461097</td>
      <td>0.455150</td>
    </tr>
    <tr>
      <th>Algeria_2002</th>
      <td>4.0</td>
      <td>2002</td>
      <td>12.0</td>
      <td>0.897816</td>
      <td>0.595164</td>
      <td>0.863495</td>
      <td>0.719969</td>
      <td>0.511700</td>
      <td>0.435257</td>
      <td>0.205148</td>
      <td>0.602964</td>
      <td>0.464119</td>
      <td>0.565523</td>
      <td>0.701248</td>
      <td>11.0</td>
      <td>Algeria_2002</td>
      <td>Algeria</td>
      <td>6.562402</td>
      <td>13.681208</td>
      <td>9.069298</td>
      <td>13.158208</td>
      <td>10.971116</td>
      <td>7.797456</td>
      <td>6.632592</td>
      <td>3.126114</td>
      <td>9.188161</td>
      <td>7.072388</td>
      <td>8.617616</td>
      <td>10.685843</td>
      <td>100.0</td>
      <td>22.893142</td>
      <td>19.992868</td>
      <td>26.839415</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>lifestyle</td>
      <td>8911.680680</td>
      <td>35.3</td>
      <td>87.843491</td>
      <td>-3.0</td>
      <td>-0.123702</td>
      <td>-0.386653</td>
      <td>-0.192044</td>
      <td>-1.171610</td>
    </tr>
    <tr>
      <th>Armenia_1997</th>
      <td>3.0</td>
      <td>1997</td>
      <td>51.0</td>
      <td>0.935000</td>
      <td>0.530000</td>
      <td>0.773000</td>
      <td>0.698000</td>
      <td>0.487000</td>
      <td>0.456500</td>
      <td>0.306500</td>
      <td>0.655500</td>
      <td>0.427500</td>
      <td>0.773500</td>
      <td>0.662000</td>
      <td>11.0</td>
      <td>Armenia_1997</td>
      <td>Armenia</td>
      <td>6.704500</td>
      <td>13.945857</td>
      <td>7.905138</td>
      <td>11.529570</td>
      <td>10.410918</td>
      <td>7.263778</td>
      <td>6.808860</td>
      <td>4.571556</td>
      <td>9.777015</td>
      <td>6.376314</td>
      <td>11.537027</td>
      <td>9.873965</td>
      <td>100.0</td>
      <td>22.962190</td>
      <td>19.740473</td>
      <td>25.475427</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>lifestyle</td>
      <td>1843.378545</td>
      <td>36.2</td>
      <td>93.895500</td>
      <td>-6.0</td>
      <td>-0.897458</td>
      <td>-0.279939</td>
      <td>0.245351</td>
      <td>-1.659638</td>
    </tr>
    <tr>
      <th>Azerbaijan_1997</th>
      <td>3.0</td>
      <td>1997</td>
      <td>31.0</td>
      <td>0.952048</td>
      <td>0.557443</td>
      <td>0.742757</td>
      <td>0.648851</td>
      <td>0.546953</td>
      <td>0.351648</td>
      <td>0.383616</td>
      <td>0.490010</td>
      <td>0.446553</td>
      <td>0.812687</td>
      <td>0.734266</td>
      <td>11.0</td>
      <td>Azerbaijan_1997</td>
      <td>Azerbaijan</td>
      <td>6.666833</td>
      <td>14.280363</td>
      <td>8.361430</td>
      <td>11.141080</td>
      <td>9.732524</td>
      <td>8.204091</td>
      <td>5.274594</td>
      <td>5.754102</td>
      <td>7.349966</td>
      <td>6.698134</td>
      <td>12.190005</td>
      <td>11.013711</td>
      <td>100.0</td>
      <td>19.322694</td>
      <td>22.319622</td>
      <td>25.421443</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>lifestyle</td>
      <td>2634.158631</td>
      <td>34.7</td>
      <td>99.470108</td>
      <td>-6.0</td>
      <td>-0.810893</td>
      <td>-0.457795</td>
      <td>0.648243</td>
      <td>-1.659638</td>
    </tr>
  </tbody>
</table>
</div>



#### Train-test split


```python
# Determine test size
job_and_indicator_df.shape
```




    (93, 45)



We have 93 rows of data. Using a 80/20% train test split gives us ~75 rows to train our model and ~18 to test


```python
# For a linear regression, we need our data to be ordinal categories (vs. disparate categories)
# We can roughly order them by making 'lifestyle' -1, 'balanced' 0 and 'achievement' 1

ordinal_mapping = pd.DataFrame([{'kmeans_value_cluster':'lifestyle','ordinal_kmeans_label': -1},
                  {'kmeans_value_cluster':'balanced','ordinal_kmeans_label': 0},
                  {'kmeans_value_cluster':'achievement','ordinal_kmeans_label': 1}])

job_and_indicator_df = pd.merge(job_and_indicator_df, ordinal_mapping, on = 'kmeans_value_cluster')
job_and_indicator_df.index = job_and_indicator_df['country_year']

job_and_indicator_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Wave</th>
      <th>Year survey</th>
      <th>Country/region</th>
      <th>Important in a job: good pay</th>
      <th>Important in a job: not too much pressure</th>
      <th>Important in a job: good job security</th>
      <th>Important in a job: a respected job</th>
      <th>Important in a job: good hours</th>
      <th>Important in a job: an opportunity to use initiative</th>
      <th>Important in a job: generous holidays</th>
      <th>Important in a job: that you can achieve something</th>
      <th>Important in a job: a responsible job</th>
      <th>Important in a job: a job that is interesting</th>
      <th>Important in a job: a job that meets one´s abilities</th>
      <th>number_of_jobs_qs_answered</th>
      <th>country_year</th>
      <th>country_name</th>
      <th>total_mentions</th>
      <th>normalized Important in a job: good pay</th>
      <th>normalized Important in a job: not too much pressure</th>
      <th>normalized Important in a job: good job security</th>
      <th>normalized Important in a job: a respected job</th>
      <th>normalized Important in a job: good hours</th>
      <th>normalized Important in a job: an opportunity to use initiative</th>
      <th>normalized Important in a job: generous holidays</th>
      <th>normalized Important in a job: that you can achieve something</th>
      <th>normalized Important in a job: a responsible job</th>
      <th>normalized Important in a job: a job that is interesting</th>
      <th>normalized Important in a job: a job that meets one´s abilities</th>
      <th>normalized total_mentions</th>
      <th>normalized achieving_responsible_initiative</th>
      <th>normalized hours_holidays_pressure</th>
      <th>normalized security_pay</th>
      <th>kmeans_labels</th>
      <th>agglom_labels</th>
      <th>dbscan_labels</th>
      <th>kmeans_value_cluster</th>
      <th>GDP per capita, PPP</th>
      <th>GINI coefficient</th>
      <th>Primary completion rate</th>
      <th>Democracy index</th>
      <th>Scaled GDP per capita, PPP</th>
      <th>Scaled GINI coefficient</th>
      <th>Scaled Primary completion rate</th>
      <th>Scaled Democracy index</th>
      <th>ordinal_kmeans_label</th>
    </tr>
    <tr>
      <th>country_year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Albania_1998</th>
      <td>3.0</td>
      <td>1998</td>
      <td>8.0</td>
      <td>0.994995</td>
      <td>0.372372</td>
      <td>0.853854</td>
      <td>0.417417</td>
      <td>0.610611</td>
      <td>0.426426</td>
      <td>0.533534</td>
      <td>0.521522</td>
      <td>0.173173</td>
      <td>0.458458</td>
      <td>0.888889</td>
      <td>11.0</td>
      <td>Albania_1998</td>
      <td>Albania</td>
      <td>6.251251</td>
      <td>15.916733</td>
      <td>5.956765</td>
      <td>13.658927</td>
      <td>6.677342</td>
      <td>9.767814</td>
      <td>6.821457</td>
      <td>8.534828</td>
      <td>8.342674</td>
      <td>2.770216</td>
      <td>7.333867</td>
      <td>14.219376</td>
      <td>100.0</td>
      <td>17.934347</td>
      <td>24.259408</td>
      <td>29.575661</td>
      <td>2</td>
      <td>0</td>
      <td>-1</td>
      <td>lifestyle</td>
      <td>3209.615372</td>
      <td>27.0</td>
      <td>92.024391</td>
      <td>5.0</td>
      <td>-0.747899</td>
      <td>-1.370790</td>
      <td>0.110121</td>
      <td>0.129798</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>Albania_2002</th>
      <td>4.0</td>
      <td>2002</td>
      <td>8.0</td>
      <td>0.952000</td>
      <td>0.455000</td>
      <td>0.812000</td>
      <td>0.645000</td>
      <td>0.550000</td>
      <td>0.397000</td>
      <td>0.483000</td>
      <td>0.513000</td>
      <td>0.254000</td>
      <td>0.411000</td>
      <td>0.294000</td>
      <td>11.0</td>
      <td>Albania_2002</td>
      <td>Albania</td>
      <td>5.766000</td>
      <td>16.510579</td>
      <td>7.891086</td>
      <td>14.082553</td>
      <td>11.186264</td>
      <td>9.538675</td>
      <td>6.885189</td>
      <td>8.376691</td>
      <td>8.896982</td>
      <td>4.405134</td>
      <td>7.127992</td>
      <td>5.098855</td>
      <td>100.0</td>
      <td>20.187305</td>
      <td>25.806452</td>
      <td>30.593132</td>
      <td>2</td>
      <td>0</td>
      <td>-1</td>
      <td>lifestyle</td>
      <td>4754.653697</td>
      <td>31.7</td>
      <td>96.880661</td>
      <td>7.0</td>
      <td>-0.578766</td>
      <td>-0.813507</td>
      <td>0.461097</td>
      <td>0.455150</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>Algeria_2002</th>
      <td>4.0</td>
      <td>2002</td>
      <td>12.0</td>
      <td>0.897816</td>
      <td>0.595164</td>
      <td>0.863495</td>
      <td>0.719969</td>
      <td>0.511700</td>
      <td>0.435257</td>
      <td>0.205148</td>
      <td>0.602964</td>
      <td>0.464119</td>
      <td>0.565523</td>
      <td>0.701248</td>
      <td>11.0</td>
      <td>Algeria_2002</td>
      <td>Algeria</td>
      <td>6.562402</td>
      <td>13.681208</td>
      <td>9.069298</td>
      <td>13.158208</td>
      <td>10.971116</td>
      <td>7.797456</td>
      <td>6.632592</td>
      <td>3.126114</td>
      <td>9.188161</td>
      <td>7.072388</td>
      <td>8.617616</td>
      <td>10.685843</td>
      <td>100.0</td>
      <td>22.893142</td>
      <td>19.992868</td>
      <td>26.839415</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>lifestyle</td>
      <td>8911.680680</td>
      <td>35.3</td>
      <td>87.843491</td>
      <td>-3.0</td>
      <td>-0.123702</td>
      <td>-0.386653</td>
      <td>-0.192044</td>
      <td>-1.171610</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>Armenia_1997</th>
      <td>3.0</td>
      <td>1997</td>
      <td>51.0</td>
      <td>0.935000</td>
      <td>0.530000</td>
      <td>0.773000</td>
      <td>0.698000</td>
      <td>0.487000</td>
      <td>0.456500</td>
      <td>0.306500</td>
      <td>0.655500</td>
      <td>0.427500</td>
      <td>0.773500</td>
      <td>0.662000</td>
      <td>11.0</td>
      <td>Armenia_1997</td>
      <td>Armenia</td>
      <td>6.704500</td>
      <td>13.945857</td>
      <td>7.905138</td>
      <td>11.529570</td>
      <td>10.410918</td>
      <td>7.263778</td>
      <td>6.808860</td>
      <td>4.571556</td>
      <td>9.777015</td>
      <td>6.376314</td>
      <td>11.537027</td>
      <td>9.873965</td>
      <td>100.0</td>
      <td>22.962190</td>
      <td>19.740473</td>
      <td>25.475427</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>lifestyle</td>
      <td>1843.378545</td>
      <td>36.2</td>
      <td>93.895500</td>
      <td>-6.0</td>
      <td>-0.897458</td>
      <td>-0.279939</td>
      <td>0.245351</td>
      <td>-1.659638</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>Azerbaijan_1997</th>
      <td>3.0</td>
      <td>1997</td>
      <td>31.0</td>
      <td>0.952048</td>
      <td>0.557443</td>
      <td>0.742757</td>
      <td>0.648851</td>
      <td>0.546953</td>
      <td>0.351648</td>
      <td>0.383616</td>
      <td>0.490010</td>
      <td>0.446553</td>
      <td>0.812687</td>
      <td>0.734266</td>
      <td>11.0</td>
      <td>Azerbaijan_1997</td>
      <td>Azerbaijan</td>
      <td>6.666833</td>
      <td>14.280363</td>
      <td>8.361430</td>
      <td>11.141080</td>
      <td>9.732524</td>
      <td>8.204091</td>
      <td>5.274594</td>
      <td>5.754102</td>
      <td>7.349966</td>
      <td>6.698134</td>
      <td>12.190005</td>
      <td>11.013711</td>
      <td>100.0</td>
      <td>19.322694</td>
      <td>22.319622</td>
      <td>25.421443</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>lifestyle</td>
      <td>2634.158631</td>
      <td>34.7</td>
      <td>99.470108</td>
      <td>-6.0</td>
      <td>-0.810893</td>
      <td>-0.457795</td>
      <td>0.648243</td>
      <td>-1.659638</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Determine test size
job_and_indicator_df.shape
```




    (93, 46)




```python
# Check nulls
job_and_indicator_df.isnull().sum().sum()
```




    0




```python
feature_list = [i for i in job_and_indicator_df.columns if i !='ordinal_kmeans_label']
target = 'ordinal_kmeans_label'
train_X, test_X, train_y, test_y = train_test_split(job_and_indicator_df[feature_list], job_and_indicator_df[target], test_size = 0.2)
```

#### Build linear regression model to predict attitudes towards jobs from development indicator values


```python
lr_achievement_axis = LinearRegression()
lr_achievement_axis.fit(train_X[indicator_features], train_X['normalized achieving_responsible_initiative'])
lr_achievement_axis.score(test_X[indicator_features], test_X['normalized achieving_responsible_initiative'])
```




    0.04441371779073655




```python
lr_lifestyle_axis = LinearRegression()
lr_lifestyle_axis.fit(train_X [indicator_features], train_X ['normalized hours_holidays_pressure'])
lr_lifestyle_axis.score(test_X[indicator_features], test_X ['normalized hours_holidays_pressure'])
```




    -0.18870168965699197




```python
lr_income_security_axis = LinearRegression()
lr_income_security_axis.fit(train_X[indicator_features], train_X['normalized security_pay'])
lr_income_security_axis.score(test_X[indicator_features], test_X['normalized security_pay'])
```




    0.03457806947824704



#### Add in cross-validation


```python
# Set number of folds for cross validation. We use CV = 10. While higher than a typical value of 3-5,  this will 
# allow us to use a higher portion of our dataset to train the model which is useful as we have a small dataset
_cv = 10 
```


```python
indicator_features
```




    ['GDP per capita, PPP',
     'GINI coefficient',
     'Primary completion rate',
     'Democracy index']




```python
for i in engineered_job_attributes:
    cvs = cross_val_score(LinearRegression(),
                                       train_X[indicator_features], 
                                       train_X[i], 
                                       cv = _cv)
    print(f'\n The mean and range of R2 values for **{i}** are {cvs.mean():.2f} and {cvs.min():.2f} - \
{cvs.max():.2f}. The values are {list(np.round(cvs, 2))}.')
```

    
     The mean and range of R2 values for **normalized achieving_responsible_initiative** are -0.01 and -0.82 - 0.58. The values are [0.41, 0.27, 0.58, 0.24, -0.82, 0.14, -0.09, -0.06, -0.52, -0.22].
    
     The mean and range of R2 values for **normalized hours_holidays_pressure** are -0.25 and -1.02 - 0.12. The values are [-0.44, 0.01, -0.24, -0.07, -0.26, -0.37, -0.14, -0.06, -1.02, 0.12].
    
     The mean and range of R2 values for **normalized security_pay** are -0.54 and -2.56 - 0.16. The values are [-0.28, 0.16, -0.03, 0.16, -0.72, -1.74, -0.01, -0.1, -2.56, -0.27].


*Overall, we see that these development indicators are not strong predictors of  attitudes towards work. Since our original indicators (GDP, GINI, education and democracy index) focus on macro features, our next set of indicators will roughly focus on more individual features. In particular we will look at connectivity (cell subscriptions per 100 people), health (life exectancy) and jobs (employment by primary secondary and tertiary sector)

#### Map in more features


```python
# cell subscriptions - https://data.worldbank.org/indicator/IT.CEL.SETS.P2?end=2016&start=1995&view=chart
# life expectancy - https://data.worldbank.org/indicator/SP.DYN.LE00.FE.IN?view=chart
# employment in ag - https://data.worldbank.org/indicator/SL.AGR.EMPL.MA.ZS?view=chart
# employment in industry - https://data.worldbank.org/indicator/SL.IND.EMPL.MA.ZS?view=chart
# employment in services - https://data.worldbank.org/indicator/SL.SRV.EMPL.MA.ZS?view=chart
```


```python
cell_subs_df = pd.read_csv("./indicator_data/cell_subscriptions.csv")
life_expectancy_female_df = pd.read_csv("./indicator_data/Life expectancy female.csv")
primary_employment_male_df = pd.read_csv("./indicator_data/Ag employment pct male.csv")
secondary_employment_male_df = pd.read_csv("./indicator_data/Industry employment pct male.csv")
tertiary_employment_male_df = pd.read_csv("./indicator_data/Services employment pct male.csv")
```


```python
# Impute nulls

new_indicator_dfs = [cell_subs_df, life_expectancy_female_df, 
                     primary_employment_male_df, secondary_employment_male_df, tertiary_employment_male_df]
new_indicator_column_names = [['cell_subs_per_100_pax'], ['life_expectancy_female'], 
                     ['primary_employment_male'], ['secondary_employment_male'], ['tertiary_employment_male']]

for i in range(0, len(new_indicator_dfs)):
    new_indicator_dfs[i].index = new_indicator_dfs[i]['Country Name']
    new_indicator_dfs[i].drop('Country Name', axis = 1, inplace=True)
    new_indicator_dfs[i] = new_indicator_dfs[i].fillna(method = 'ffill', axis=1)             # first try a forward fill (i.e., base on the last available value)
    new_indicator_dfs[i] = new_indicator_dfs[i].fillna(method = 'backfill', axis=1)          # next try a backward fill (i.e., base on the next available value)
    new_indicator_dfs[i] = new_indicator_dfs[i].fillna(value = new_indicator_dfs[i].mean())  # finally impute with df mean
```


```python
# Match World Bank country names to World Values Survey Country names

country_name_map_df = pd.read_csv("./indicator_data/WB to WSV country name.csv", index_col = 'Country Name')

for i in range(0, len(new_indicator_dfs)):
    new_indicator_dfs[i] = pd.concat([new_indicator_dfs[i], country_name_map_df], axis = 1)
```


```python
# Flatten data

for i in range(0, len(new_indicator_dfs)):
    new_indicator_dfs[i].index = new_indicator_dfs[i]['WVS country name']
    new_indicator_dfs[i] = pd.DataFrame(new_indicator_dfs[i].stack(), columns=new_indicator_column_names[i])
    new_indicator_dfs[i].reset_index(inplace=True)
    new_indicator_dfs[i].index = new_indicator_dfs[i]['WVS country name'] + "_" + new_indicator_dfs[i]['level_1']
    new_indicator_dfs[i].drop(['WVS country name', 'level_1'], axis = 1, inplace= True)
```


```python
# Reassign dataframe names

cell_subs_df                 = new_indicator_dfs[0]
life_expectancy_female_df    = new_indicator_dfs[1]
primary_employment_male_df   = new_indicator_dfs[2]
secondary_employment_male_df = new_indicator_dfs[3]
tertiary_employment_male_df  = new_indicator_dfs[4]
```


```python
# Merge new indicators into 'jobs_and_indicators_df

job_and_indicator_df=pd.merge(job_and_indicator_df, cell_subs_df, left_index=True, right_index=True, how = 'left')
job_and_indicator_df=pd.merge(job_and_indicator_df, life_expectancy_female_df, left_index=True, right_index=True, how = 'left')
job_and_indicator_df=pd.merge(job_and_indicator_df, primary_employment_male_df, left_index=True, right_index=True, how = 'left')
job_and_indicator_df=pd.merge(job_and_indicator_df, secondary_employment_male_df, left_index=True, right_index=True, how = 'left')
job_and_indicator_df=pd.merge(job_and_indicator_df, tertiary_employment_male_df, left_index=True, right_index=True, how = 'left')
```


```python
# Check nulls and shape
job_and_indicator_df.isnull().sum().sum()
```




    0




```python
job_and_indicator_df.shape
```




    (93, 51)



#### Build linear regression including updated economic indicators


```python
# Train test split new dataframe (including new indicator features)
feature_list = [i for i in job_and_indicator_df.columns if i !='ordinal_kmeans_label']
target = 'ordinal_kmeans_label'
train_X, test_X, train_y, test_y = train_test_split(job_and_indicator_df[feature_list], job_and_indicator_df[target], test_size = 0.2)
```


```python
# Define list of indicators (features for new model)
full_indicator_features = ['GDP per capita, PPP', 'GINI coefficient', 'Primary completion rate', 'Democracy index', 
                           'cell_subs_per_100_pax', 'life_expectancy_female', 'primary_employment_male', 
                           'secondary_employment_male', 'tertiary_employment_male']
```


```python
for i in engineered_job_attributes:
    cvs = cross_val_score(LinearRegression(),
                                       train_X[full_indicator_features], 
                                       train_X[i], 
                                       cv = _cv)
    print(f'\n The mean and range of R2 values for **{i}** are {cvs.mean():.2f} and {cvs.min():.2f} - \
{cvs.max():.2f}. The values are {list(np.round(cvs, 2))}.')
```

    
     The mean and range of R2 values for **normalized achieving_responsible_initiative** are -0.09 and -0.72 - 0.50. The values are [-0.14, -0.23, 0.31, 0.02, -0.72, -0.65, -0.05, -0.01, 0.5, 0.1].
    
     The mean and range of R2 values for **normalized hours_holidays_pressure** are -0.65 and -3.73 - 0.31. The values are [-0.03, 0.16, -1.11, -0.68, -0.04, -3.73, 0.31, -0.78, 0.31, -0.92].
    
     The mean and range of R2 values for **normalized security_pay** are -0.46 and -3.16 - 0.41. The values are [-0.61, 0.33, -0.18, 0.36, 0.41, -0.67, 0.32, -3.16, -0.41, -0.98].


*Both our original regression with four economic indicators and our expanded regression with an additional five indicators have low R2 values. Therefore, these variables alone are not powerful predictors of a country's values system and attitude towards work.*

*There could be several reasons for this finding:*
- Socio-economic indicators are overpowered by historical culture (e.g., former Soviet countries may all share a common cultural background). Adding variables that encode this information (e.g., geographic region) could control for this potentially confounding variable, providing visibility into the more nuanced effects of other factors (e.g., GDP)
- Values and attitudes are very difficult to measure, exhibiting heterogeneity within as well as between countries. Our relatively small data set of ~100 country/ year combinations provides limited room for error. Additional waves of the World Values Survey, which will come online as soon as 2018, will provide a more robust and comprehensive dataset.

#### Following the thread of socio-economic indicators being overpowered by historical culture, we will run a regression including geographic dummies to control for a country's region


```python
# Map in region
```


```python
_region_dictionary = pd.DataFrame(region_dictionary)
```


```python
job_and_indicator_df = pd.merge(job_and_indicator_df, _region_dictionary, on = 'country_name', how = 'left')
job_and_indicator_df = job_and_indicator_df.set_index('country_year')
```


```python
# Get dummies
```


```python
job_and_indicator_df = pd.concat([job_and_indicator_df, pd.get_dummies(job_and_indicator_df['region'], prefix='region')],axis = 1)
```


```python
job_and_indicator_df.shape
```




    (93, 61)




```python
assert job_and_indicator_df.isnull().sum().sum() == 0
```


```python
# Train test split new dataframe (including new region labels)
```


```python
feature_list = [i for i in job_and_indicator_df.columns if i !='ordinal_kmeans_label']
target = 'ordinal_kmeans_label'
train_X, test_X, train_y, test_y = train_test_split(job_and_indicator_df[feature_list], job_and_indicator_df[target], test_size = 0.2)
```


```python
# Determine feature columns
```


```python
region_dummies = ['region_ Asia', 'region_ South America', 'region_Africa', 'region_Asia',
                  'region_Europe', 'region_MENA', 'region_North America',
                  'region_Oceania', 'region_South America', 'region_USSR+']
```


```python
indicator_and_region_columns = full_indicator_features + region_dummies
```


```python
for i in engineered_job_attributes:
    cvs = cross_val_score(LinearRegression(),
                                       train_X[indicator_and_region_columns], 
                                       train_X[i], 
                                       cv = _cv)
    print(f'\n The mean and range of R2 values for **{i}** are {cvs.mean():.2f} and {cvs.min():.2f} - \
{cvs.max():.2f}. The values are {list(np.round(cvs, 2))}.')
```

    
     The mean and range of R2 values for **normalized achieving_responsible_initiative** are -0.59 and -3.82 - 0.65. The values are [-0.07, 0.13, 0.04, 0.57, -3.82, -1.84, -0.97, -0.63, 0.08, 0.65].
    
     The mean and range of R2 values for **normalized hours_holidays_pressure** are -0.63 and -4.01 - 0.36. The values are [0.25, -1.53, 0.15, -0.89, -4.01, 0.36, -0.14, -0.29, -0.23, 0.01].
    
     The mean and range of R2 values for **normalized security_pay** are -0.73 and -2.15 - 0.39. The values are [-0.36, -1.3, -1.03, 0.26, 0.39, -2.15, -1.63, -0.68, 0.03, -0.8].


*In this regression, adding in region dummies does not increase the R2 score. Given our training data of 73 rows and feature count of close to 20, we may be approaching overfitting. A next step could be to do regularized regression (e.g., LASSO, ridge)*

*For now, given that values are challenging to predict, we will instead build on our clustering analysis. *
