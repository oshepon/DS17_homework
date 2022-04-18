import pandas as pd
import numpy as np
import math
from sklearn.datasets import make_circles, make_moons
from IPython.display import display, clear_output
import plotly_express as px
from plotly.subplots import make_subplots

# import cufflinks as cf; cf.go_offline()
# import plotly_express as px
# import ipywidgets as widgets

#create the datasets according to input parameters
# returned df has col 'x', 'y' and target: col 'label' with categories A or B
def creat_dataset(n_smpl_size=1000, noise_lvl=0.2, type='moons'):
    if type == 'moons':
        points, label = make_moons(n_samples=n_smpl_size, noise=noise_lvl)
    elif type == 'circles':
        points, label = make_circles(n_samples=n_smpl_size, noise=noise_lvl)
    else:
        print('function creat_dataset(): type should be "moons" or "circles"')
        return None
    df = pd.DataFrame(points, columns=['x','y'])
    df['label'] = label
    df.label = df.label.map({0:'A', 1:'B'})
    return df

#calculates accuracy given df, col with true data vs predicted
def accuracy(true_col, pred_col):
#     hit_lst = df.apply(lambda x: 1 if x[true_col] == x[pred_col] else 0, axis=1)
    hit_lst = [1 if a == b else 0 for a, b in zip(true_col, pred_col)]
#    hits = hit_lst.sum()
    hits = sum(hit_lst)
    acc = hits/len(hit_lst)
    #print(f'test accuracy={acc :.2f}')
    return acc


# used in Q5
#plot TRE and TESTE vs model (log) paramter per dataset
def plot_CV_train_test_error(df, n, nl, data_type, clf_type):
    ds_rdata = df.query('n == @n and noise_lvl == @nl and data_type == @data_type and model == @clf_type')
    fig = px.line(ds_rdata, x='model_param', y=['TRE', 'TESTE'], log_x= True)
    line_1 = f'Test Accuracy by <b>{clf_type}</b> Hyperparamter <br>'
    line_2 = f'         Dataset: n= {n}, noise= {nl}, data_type= {data_type}'
    fig.update_layout(title_text= line_1 + line_2)
    fig.update_yaxes(title_text= 'Accuracy', secondary_y=False)
    return fig


# used in Q5
#plot DIFF_E vs model (log) paramter per dataset
def plot_CV_DIFF_E(df, n, nl, data_type, clf_type):
    ds_rdata = df.query('n == @n and noise_lvl == @nl and data_type == @data_type and model == @clf_type')
    fig = px.line(ds_rdata, x='model_param', y=['E_DIFF'], log_x= True, color_discrete_sequence=px.colors.qualitative.Alphabet)
    line_1 = f'Train-Val Delta Accuracy by <b>{clf_type}</b> Hyperparamter <br>'
    line_2 = f'         Dataset: n= {n}, noise= {nl}, data_type= {data_type}'
    fig.update_layout(title_text= line_1 + line_2)
    fig.update_yaxes(title_text= 'Accuracy Delta', secondary_y= False)
    return fig

# #plot DIFF_E vs model (log) paramter per dataset
# def plot_CV_DIFF_vs_n(df, model_param, nl, data_type, clf_type):
#     ds_rdata = df.query('noise_lvl == @nl and data_type == @data_type and model == @clf_type and model_param == @model_param')
#     fig = px.line(ds_rdata, x='n', y=['E_DIFF'], log_x= True, color_discrete_sequence=px.colors.qualitative.Alphabet)
#     line_1 = f'Train-Val Delta Accuracy by <b>{clf_type}</b> n Sample Size <br>'
#     line_2 = f'         Dataset: noise= {nl}, data_type= {data_type}, model_param= {model_param}'
#     fig.update_layout(title_text= line_1 + line_2)
#     fig.update_yaxes(title_text= 'Accuracy Delta', secondary_y= False)
#     return fig


#used in Q4
#plot DIFF_E vs model (log) paramter per dataset
def plot_CV_DIFF_vs_n2(df, model_param, nl, data_type, clf_type):
    ds_rdata = df.query('noise_lvl == @nl and data_type == @data_type and model == @clf_type and model_param == @model_param')
    subfig = make_subplots(specs=[[{"secondary_y": True}]])
    fig = px.line(ds_rdata, x='n', y=['E_DIFF'], log_x= False, color_discrete_sequence=px.colors.qualitative.Alphabet, render_mode="webgl",)
    fig2 = px.line(ds_rdata, x='n', y=['TRE', 'TESTE'], log_x= False, render_mode="webgl",)
    fig2.update_traces(yaxis="y2")
    subfig.add_traces(fig.data + fig2.data)
    subfig.layout.xaxis.title="n Sample Size"
    subfig.layout.xaxis.type="log"
    subfig.layout.yaxis.title="Accuracy Delta"
    subfig.layout.yaxis2.title="Accuracy"
    line_1 = f'Train-Val Delta Accuracy by <b>{clf_type}</b> n Sample Size <br>'
    line_2 = f'         Dataset: noise= {nl}, data_type= {data_type}, model_param= {model_param}'
    subfig.update_layout(title_text= line_1 + line_2)
    return subfig

# #plot E_DIFF vs model (log) paramter per dataset
# #plot_CV_DIFF_vs_n3(smallest_E_df, nl, data_type)
# def plot_CV_DIFF_vs_n3(df, nl, data_type):
#     ds_rdata_svm = df.query('noise_lvl == @nl and data_type == @data_type and model == "svm"')
#     ds_rdata_svm = ds_rdata_svm.rename(columns={'E_DIFF': 'SVM_E_DIFF'})
#     ds_rdata_log = df.query('noise_lvl == @nl and data_type == @data_type and model == "logistic"')
#     ds_rdata_log = ds_rdata_log.rename(columns={"E_DIFF": "Logistic_E_DIFF"})
#     subfig = make_subplots(specs=[[{"secondary_y": False}]])
#     fig = px.line(ds_rdata_svm, x='n', y=['SVM_E_DIFF'], log_x= True, color_discrete_sequence=px.colors.qualitative.Alphabet, render_mode="webgl",)
#     fig2 = px.line(ds_rdata_log, x='n', y=['Logistic_E_DIFF'], log_x= True, render_mode="webgl",)
#     #fig2.update_traces(yaxis="y2")
#     subfig.add_traces(fig.data + fig2.data)
#     subfig.layout.xaxis.title="n Sample Size"
#     subfig.layout.xaxis.type="log"
#     subfig.layout.yaxis.title="Accuracy Delta"
#     #subfig.layout.yaxis2.title="Accuracy"
#     line_1 = f'Models Train-Val Delta Accuracy (E_DIFF) by n Sample Size <br>'
#     line_2 = f'         Dataset: noise= {nl}, data_type= {data_type}'
#     subfig.update_layout(title_text= line_1 + line_2)
#     return subfig

# #
# def plot_CV_DIFF_vs_n4(df, nl, data_type):
#     ds_rdata_svm = df.query('noise_lvl == @nl and data_type == @data_type and model == "svm"')
#     ds_rdata_svm = ds_rdata_svm.rename(columns={'TESTE': 'SVM_TESTE'})
#     ds_rdata_log = df.query('noise_lvl == @nl and data_type == @data_type and model == "logistic"')
#     ds_rdata_log = ds_rdata_log.rename(columns={"TESTE": "Logistic_TESTE"})
#     subfig = make_subplots(specs=[[{"secondary_y": False}]])
#     fig = px.line(ds_rdata_svm, x='n', y=['SVM_TESTE'], log_x= True, color_discrete_sequence=px.colors.qualitative.Alphabet, render_mode="webgl",)
#     fig2 = px.line(ds_rdata_log, x='n', y=['Logistic_TESTE'], log_x= True, render_mode="webgl",)
#     #fig2.update_traces(yaxis="y2")
#     subfig.add_traces(fig.data + fig2.data)
#     subfig.layout.xaxis.title="n Sample Size"
#     subfig.layout.xaxis.type="log"
#     subfig.layout.yaxis.title="Accuracy"
#     #subfig.layout.yaxis2.title="Accuracy"
#     line_1 = f'Models Train-Val Accuracy (TESTE) by n Sample Size <br>'
#     line_2 = f'         Dataset: noise= {nl}, data_type= {data_type}'
#     subfig.update_layout(title_text= line_1 + line_2)
#     return subfig

#per data type and model plot Accuracy of each noise level (vs n sample size)
# def plot_CV_DIFF_vs_n5(df, data_type, clf_type):
#     ds_rdata = df.query('data_type == @data_type and model == @clf_type')
#     table = pd.pivot_table(ds_rdata, values='TESTE', index=['n'], columns=['noise_lvl'], aggfunc=np.mean).reset_index()
#     table = table.rename(columns={0.0: "nl 0.0", 0.1: "nl 0.1", 0.2: "nl 0.2", 0.3: "nl 0.3", 0.4: "nl 0.4", 0.5: "nl 0.5"})
#     fig = px.line(table, x='n', y=["nl 0.0", "nl 0.1", "nl 0.2", "nl 0.3", "nl 0.4", "nl 0.5"], log_x= True)
#     fig.update_layout(title_text= f'Test Accuracy by <b>{clf_type}</b> for data_type <b>{data_type}</b>')
#     fig.update_yaxes(title_text= 'Accuracy', secondary_y=False)
#     return fig


# #per data type and model plot Accuracy of each noise level (vs n sample size)
# def plot_CV_DIFF_vs_n5(df, data_type, clf_type):
#     ds_rdata = df.query('data_type == @data_type and model == @clf_type')
#     fig = px.line(ds_rdata, x='n', y='TESTE', color= 'noise_lvl', log_x= True)
#     fig.update_layout(title_text= f'Test Accuracy by <b>{clf_type}</b> for data_type <b>{data_type}</b>')
#     fig.update_yaxes(title_text= 'Accuracy', secondary_y=False)
#     return fig


# #per data type and model plot Accuracy of each noise level (vs n sample size)
# def plot_CV_DIFF_vs_n5(df, data_type, clf_type):
#     ds_rdata = df.query('data_type == @data_type and model == @clf_type')
#     subfig = make_subplots(specs=[[{"secondary_y": True}]])
#     fig = px.line(ds_rdata, x='n', y=['E_DIFF'], color = 'noise_lvl', render_mode="webgl",)
#     fig2 = px.line(ds_rdata, x='n', y=['TESTE'], color = 'noise_lvl', render_mode="webgl",color_discrete_sequence=px.colors.qualitative.Alphabet,)
#     fig2.update_traces(yaxis="y2", showlegend = False, )
#     subfig.add_traces(fig.data + fig2.data)
#     subfig.layout.xaxis.title="n Sample Size"
#     subfig.layout.xaxis.type="log"
#     subfig.layout.yaxis.title="Accuracy Delta"
#     subfig.layout.yaxis2.title="Accuracy"
#     line_1 = f'Accuracy and Error Difference by <b>{clf_type}</b> Model vs n Sample Size <br>'
#     line_2 = f'         Dataset: data_type= {data_type}'
#     subfig.update_layout(title_text= line_1 + line_2, )
#     return subfig


#used in Q6
#per data type and model plot Accuracy of each noise level (vs n sample size)
def plot_CV_DIFF_vs_n6(df, data_type, clf_type, gamma, c):
    ds_rdata = subset_df(df, data_type,clf_type, gamma, c)
    subfig = make_subplots(specs=[[{"secondary_y": True}]])
    fig = px.line(ds_rdata, x='n', y=['E_DIFF'], color = 'noise_lvl', render_mode="webgl",)
    fig2 = px.line(ds_rdata, x='n', y=['TESTE'], color = 'noise_lvl', render_mode="webgl",color_discrete_sequence=px.colors.qualitative.Alphabet,)
    fig2.update_traces(yaxis="y2", showlegend = False, )
    subfig.add_traces(fig.data + fig2.data)
    subfig.layout.xaxis.title="n Sample Size"
    subfig.layout.xaxis.type="log"
    subfig.layout.yaxis.title="Accuracy Delta"
    subfig.layout.yaxis2.title="Accuracy"
    line_1 = f'Accuracy vs n Sample Size <br>'
    model_param = gamma if clf_type == 'svm' else c
    line_2 = f'         Dataset: data_type= {data_type} by <b>{clf_type}</b> model_param= {model_param}'
    subfig.update_layout(title_text= line_1 + line_2, )
    return subfig


#used in Q6
#per data type and model plot Accuracy of each noise level (vs n sample size)
def plot_CV_DIFF_vs_prop(df, data_type, clf_type, gamma, c):
    ds_rdata = subset_df(df, data_type,clf_type, gamma, c)
    #ds_rdata['TESTE_P'] = (ds_rdata['TESTE'] / ds_rdata['TESTE'].sum()) * 100
    #ds_rdata['TESTE_P'] = ds_rdata['TESTE'] / ds_rdata['TESTE'].max()
    #ds_rdata['E_DIFF_P'] = ds_rdata['E_DIFF'] / ds_rdata['E_DIFF'].min()
    #X = train.iloc[:,:-1] #without label
    #y = train.iloc[:,-1]  #only label
    pd.options.mode.chained_assignment = None  # default='warn'
    test_max = ds_rdata['TESTE'].max()
    ds_rdata['TESTE_P'] = ds_rdata['TESTE'].apply(lambda x: x/test_max)
    subfig = make_subplots(specs=[[{"secondary_y": True}]])
    fig = px.line(ds_rdata, x='n', y=['E_DIFF'], color = 'noise_lvl', render_mode="webgl",)
    fig2 = px.line(ds_rdata, x='n', y=['TESTE_P'], color = 'noise_lvl', render_mode="webgl",color_discrete_sequence=px.colors.qualitative.Alphabet,)
    fig2.update_traces(yaxis="y2", showlegend = True)###False, )
    subfig.add_traces(fig2.data)### + fig.data)
    subfig.layout.xaxis.title="n Sample Size"
    subfig.layout.xaxis.type="log"
    subfig.layout.yaxis.title="Accuracy Delta"
    subfig.layout.yaxis2.title="Accuracy/Best-Accuracy"
    line_1 = f'Accuracy Proportion to Best Accuracy per Model/Hyperparameter vs n Sample Size <br>'
    model_param = gamma if clf_type == 'svm' else c
    line_2 = f'         Dataset: data_type= {data_type} by <b>{clf_type}</b> model_param= {model_param}'
    subfig.update_layout(title_text= line_1 + line_2, )
    return subfig



#return subset of df based on params
def subset_df(df, data_type, clf_type, gamma, c):
    if clf_type == 'svm':
        param = gamma
    elif clf_type == 'logistic':
        param = c
    x = df.query('data_type == @data_type and model == @clf_type and model_param == @param')
    return x


# used in Q1
#plot accuracy mean and std(error) by model param and find highest acc
def plot_acc_vs_gamma(rdata_df, cmodel, cn, data_type):
    subset = rdata_df.query('n == @cn and model == @cmodel and data_type == @data_type')
    gamma_acc_means = subset.groupby(by='model_param')[['TESTE', 'TESTE_VAR']].mean()
    gamma_acc_means['TESTE_POOL_STD'] = gamma_acc_means.TESTE_VAR.apply(lambda x: math.sqrt(x)) #sqrt of pooled variance
    #plot
    fig = px.line(gamma_acc_means,y='TESTE', error_y='TESTE_POOL_STD', log_x= True)
    line_1 = f'Accuracy Mean of combined different Noise Levels for <b>{data_type} </b> <br>'
    line_2 = f'            Dataset: n= {cn}, Model= {cmodel}'
    fig.update_layout(title_text= line_1 + line_2, hovermode="x unified")
    fig.update_traces(showlegend=True,  name='acc val mean')
    fig.update_yaxes(title_text= 'Accuracy', secondary_y=False)
    #params for max acc
    gamma_max = gamma_acc_means.TESTE.idxmax()
    acc_max = gamma_acc_means.TESTE.max()
    gamma_max_std = gamma_acc_means.TESTE_POOL_STD.loc[gamma_max]
    return fig, gamma_max, acc_max, gamma_max_std


# used in Q2
#plot accuracy mean and std by model param and find lowest std
def plot_std_vs_gamma(rdata_df, cmodel, cn, data_type):
    subset = rdata_df.query('n == @cn and model == @cmodel and data_type == @data_type')
    gamma_acc_means = subset.groupby(by='model_param')[['TESTE', 'TESTE_VAR']].mean()
    gamma_acc_means['TESTE_POOL_STD'] = gamma_acc_means.TESTE_VAR.apply(lambda x: math.sqrt(x)) #sqrt of pooled variance
    #plot from 2 subplots
    subfig = make_subplots(specs=[[{"secondary_y": True}]])
    fig1 = px.line(gamma_acc_means,y='TESTE_POOL_STD', color_discrete_sequence=px.colors.qualitative.Alphabet,
                   render_mode="webgl", )
    fig1.update_traces(showlegend=True, name='acc val std')
    fig2 = px.line(gamma_acc_means,y='TESTE', render_mode="webgl", )
    fig2.update_traces(yaxis="y2",showlegend=True,  name='acc val mean')
    subfig.layout.xaxis.type="log"
    subfig.add_traces(fig1.data + fig2.data)
    subfig.layout.xaxis.title="model parameter"
    subfig.layout.yaxis.title="Accuracy STD"
    subfig.layout.yaxis2.title="Accuracy"
    line_1 = f'Validation Delta Accuracy and Mean Accuracy of combined different Noise Levels for <b>{data_type} </b> <br>'
    line_2 = f'            Dataset: n= {cn}, Model= {cmodel}' 
    subfig.update_layout(title_text= line_1 + line_2, hovermode="x unified")
    #params for min acc std
    gamma_min_std = gamma_acc_means.TESTE_POOL_STD.idxmin()
    std_min = gamma_acc_means.TESTE_POOL_STD.min()
    acc_std_min = gamma_acc_means.TESTE.loc[gamma_min_std]
    return subfig, gamma_min_std, std_min, acc_std_min


# used in Q3
#plot accuracy mean and std by model param and find lowest std
def plot_acc_vs_clog(rdata_df, cmodel, data_type):
    subset = rdata_df.query('model == @cmodel and data_type == @data_type')
    clog_acc = subset.groupby(by=['model_param', 'n'])[['TESTE', 'TESTE_VAR']].mean().reset_index()
    clog_acc['TESTE_POOL_STD'] = clog_acc.TESTE_VAR.apply(lambda x: math.sqrt(x)) #sqrt of pooled variance (pooled std)
    line1 = f'Accuracy of Sample Sizes by <b>{cmodel}</b> C parameter of combined different Noise Levels <br>'
    line2 = f'        for <b>{data_type} </b>'
    fig = px.line(clog_acc, x= "model_param", y="TESTE", color="n", log_x= True, #error_y='TESTE_POOL_STD',
                  #line_shape="spline", render_mode="svg",
                  color_discrete_sequence=px.colors.qualitative.G10,
                  title= line1 + line2)
    #fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title_text= 'Accuracy', secondary_y=False)
    return fig

import statistics
from scipy.stats import gmean
def df_acc_vs_gamma(rdata_df, cmodel, cn, data_type):
    subset = rdata_df.query('n == @cn and model == @cmodel and data_type == @data_type')
    gamma_acc_means = subset.groupby(by='model_param').agg({'TESTE': [np.mean, np.median, statistics.harmonic_mean, gmean, np.min, np.max, np.std, np.var]}).round(3)
    return gamma_acc_means

# used in Q3
#plot accuracy mean and std by model param and find lowest std
def plot_acc_std_vs_clog(rdata_df, cmodel, data_type):
    subset = rdata_df.query('model == @cmodel and data_type == @data_type')
    clog_acc = subset.groupby(by=['model_param', 'n'])[['TESTE', 'TESTE_VAR']].mean().reset_index()
    clog_acc['TESTE_POOL_STD'] = clog_acc.TESTE_VAR.apply(lambda x: math.sqrt(x)) #sqrt of pooled variance
    line1 = f'Accuracy <b>STD</b> of Sample Sizes by <b>{cmodel}</b> C parameter of combined different Noise Levels <br>'
    line2 = f'        for <b>{data_type} </b>'
    fig = px.line(clog_acc, x= "model_param", y="TESTE_POOL_STD", color="n", log_x= True, #error_y='TESTE_POOL_STD',
                  #line_shape="spline", render_mode="svg",
                  color_discrete_sequence=px.colors.qualitative.G10,
                  title= line1 + line2)
    #fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title_text= 'Accuracy STD', secondary_y=False)
    return fig

