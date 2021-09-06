import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
def optim_plot(cur_code):
    import FX_DATA_GEN as DG
    cntry = DG.Get_country_name(cur_code)
    path = './log/%s/CLXAMSGrad_LSTMClx_bs{32}_lr{1.0e-05}_eps{1.0e-03}_ech{400}/history.pkl'% (cur_code)
    df_amsgrad = pkl.load(open(path, 'rb'))

    path = './log/%s/CLXADAM_LSTMClx_bs{32}_lr{1.0e-05}_eps{1.0e-03}_ech{400}/history.pkl'% (cur_code)
    df_adam = pkl.load(open(path, 'rb'))

    path = './log/%s/CLXRMSProp_LSTMClx_bs{32}_lr{1.0e-05}_eps{1.0e-03}_ech{400}/history.pkl'% (cur_code)
    df_rms = pkl.load(open(path, 'rb'))

    path = './log/%s/CLXTHEOPOULA_LSTMClx_bs{32}_lr{1.0e-05}_ech{400}_eta{0.0e+00}_beta{1.0e+15}_r{0}_eps{1.0e-03}/history.pkl'% (cur_code)
    df_theopoula = pkl.load(open(path, 'rb'))
    
    fig, ax = plt.subplots(nrows = 4, ncols = 1, figsize=(9,10))
    for i, key in zip(range(1, 5), df_theopoula.keys()):
        ax[i-1].plot(df_amsgrad[key], label='AMSGRAD')
        ax[i-1].plot(df_adam[key], label='ADAM')
        ax[i-1].plot(df_rms[key], label='RMSProp')
        ax[i-1].plot(df_theopoula[key], label='TheoPouLa')
        ax[i-1].set_title(key)
        ax[i-1].set_xlabel('Epoch')
        if key == 'training_loss' or key == 'test_loss':
            ax[i-1].set_ylabel('Loss')
        else:
            ax[i-1].set_ylabel('Accuracy Ratio)')
    h1, l1 = ax[0].get_legend_handles_labels()
    fig.legend(h1, l1, loc='lower right', ncol=2, fontsize=9)
    fig.suptitle(f'{cntry}: Test vs Train Loss & Accuracy:\nOptimiser algorithm comparisons', fontsize=15)
    fig.tight_layout() 
    print('AMSGrad:  test_loss -',np.array(df_amsgrad['test_loss']).min().item())
    print('ADAM:  test_loss -',np.array(df_adam['test_loss']).min().item())
    print('RMSprop:  test_loss -',np.array(df_rms['test_loss']).min().item())
    print('THEOPOULA:  test_loss -',np.array(df_theopoula['test_loss']).min().item())
    
    thisdict = {
        "Optimizer": ["AMSGrad","ADAM","RMSProp","THEOPOULA"],
        "min_test_loss": [np.array(df_amsgrad['test_loss']).min().item(),
                        np.array(df_adam['test_loss']).min().item(),
                        np.array(df_rms['test_loss']).min().item(),
                        np.array(df_theopoula['test_loss']).min().item()],
        "mean_test_loss": [np.array(df_amsgrad['test_loss']).mean().item(),
                        np.array(df_adam['test_loss']).mean().item(),
                        np.array(df_rms['test_loss']).mean().item(),
                        np.array(df_theopoula['test_loss']).mean().item()]
                }
    path = f'./log/{cur_code}'
    pkl.dump(thisdict, open(path+'/optim_compare.pkl', 'wb'))
    plt.show()


def prob_plot(results):
    dataset_to_plot = results.copy()
    labels = ['High Crash Confidence','Medium Crash Confidence','Low Crash Confidence','Small deppreciation','Small Appreciation','Large Appreciation']

    #melt_probs = pd.melt(dataset_to_plot[['High Crash Confidence','Medium Crash Confidence','Low Crash Confidence','Small deppreciation','Large Appreciation']].reset_index(),
    #                    value_vars=['High Crash Confidence','Medium Crash Confidence','Low Crash Confidence','Small deppreciation','Large Appreciation'],
    #                    id_vars = 'Date', var_name = 'Classification', value_name='Probabilities')
    #melt_probs.sort_values(by=['Date', 'Classification'], inplace=True)


    col_pallet = ['#f20000','#da00f2','#3000f2','#0089f2','#00f2c5','#04f200']
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(16,4))

    ax[0].stackplot(dataset_to_plot.index,
                dataset_to_plot['High Crash Confidence'].tolist(),
                dataset_to_plot['Medium Crash Confidence'].tolist(),
                dataset_to_plot['Low Crash Confidence'].tolist(),
                dataset_to_plot['Small deppreciation'].tolist(),
                dataset_to_plot['Small Appreciation'].tolist(),
                dataset_to_plot['Large Appreciation'].tolist(),
                labels=labels,colors=col_pallet)
    ax[0].set_xlabel('Time')
    ax[0].tick_params(axis='x', rotation=45)
    ax[0].set_ylabel('Probabilities')
    #ax[0].legend()
    ax[1].plot(dataset_to_plot.index, dataset_to_plot['Predicted_prob'], label='Probability of Predicted Classification')
    ax[1].plot(dataset_to_plot.index, dataset_to_plot['Correct_prob'], label='Probability of Correct Classification ')
    #ax[1].legend(loc='upper right')
    ax[1].tick_params(axis='x', rotation=45)
    #fig.suptitle('Probability visualisation: Model Resulting Probabilities Predictions', fontsize=15)
    h1, l1 = ax[0].get_legend_handles_labels()
    h2, l2 = ax[1].get_legend_handles_labels()
    fig.legend(h1, l1, loc='upper left', ncol=2, fontsize=9)
    fig.legend(h2, l2, loc='upper right', ncol=1, fontsize=9)
    plt.show()

def multi_probs_plot(cur_code):
    import FX_DATA_GEN as DG
    
    cntry = DG.Get_country_name(cur_code)
    path = './log/%s/CLXAMSGrad_LSTMClx_bs{32}_lr{1.0e-05}_eps{1.0e-03}_ech{400}/results.pkl'% (cur_code)
    df_amsgrad = pkl.load(open(path, 'rb'))
    df_amsgrad = pd.DataFrame(df_amsgrad)
    
    path = './log/%s/CLXADAM_LSTMClx_bs{32}_lr{1.0e-05}_eps{1.0e-03}_ech{400}/results.pkl'% (cur_code)
    df_adam = pkl.load(open(path, 'rb'))
    df_adam = pd.DataFrame(df_adam)
    
    path = './log/%s/CLXRMSProp_LSTMClx_bs{32}_lr{1.0e-05}_eps{1.0e-03}_ech{400}/results.pkl'% (cur_code)
    df_rms = pkl.load(open(path, 'rb'))
    df_rms = pd.DataFrame(df_rms)
    
    path = './log/%s/CLXTHEOPOULA_LSTMClx_bs{32}_lr{1.0e-05}_ech{400}_eta{0.0e+00}_beta{1.0e+15}_r{0}_eps{1.0e-03}/results.pkl'% (cur_code)
    df_theopoula = pkl.load(open(path, 'rb'))    
    df_theopoula = pd.DataFrame(df_theopoula)
    
    dataframes = [df_amsgrad, df_adam, df_rms, df_theopoula]
    optimisers = ['AMSGrad', 'ADAM', 'RMSProp', 'THEO_POULA']
    # Prepare for plotting 
    labels = ['High Crash Confidence','Medium Crash Confidence','Low Crash Confidence','Small deppreciation','Small Appreciation','Large Appreciation']
    col_pallet = ['#f20000','#da00f2','#3000f2','#0089f2','#00f2c5','#04f200']
    fig, ax = plt.subplots(nrows = len(dataframes), ncols = 2, figsize=(20,12))
    for i in range(len(dataframes)):
        dataset_to_plot = dataframes[i].copy()
        ax[0,0].set_title('Probability results of all classifications ')
        ax[0,1].set_title('Probability of Correct classification vs Predicted classification')
        ax[i,0].stackplot(dataset_to_plot.index,dataset_to_plot['High Crash Confidence'].tolist(),
                     dataset_to_plot['Medium Crash Confidence'].tolist(),
                     dataset_to_plot['Low Crash Confidence'].tolist(),
                     dataset_to_plot['Small deppreciation'].tolist(),
                     dataset_to_plot['Small Appreciation'].tolist(),
                     dataset_to_plot['Large Appreciation'].tolist(),
                     labels=labels,colors=col_pallet)
        ax[i,0].set_xlabel('Time')
        ax[i,0].tick_params(axis='x', rotation=45)
        ax[i,0].set_ylabel(f'{optimisers[i]}:All Probabilities', rotation = 90)
        ax[i,1].plot(dataset_to_plot.index, dataset_to_plot['Predicted_prob'], label='Probability of Predicted Classification')
        ax[i,1].plot(dataset_to_plot.index, dataset_to_plot['Correct_prob'], label='Probability of Correct Classification ')
        ax[i,1].tick_params(axis='x', rotation=45)
        ax[i,1].set_ylabel(f'{optimisers[i]}: Probability', rotation = 90)
    fig.suptitle(f'{cntry}:Probability visualisation: Model Resulting Probabilities Predictions', fontsize=15)
    h1, l1 = ax[i,0].get_legend_handles_labels()
    h2, l2 = ax[i,1].get_legend_handles_labels()
    fig.legend(h1+h2, l1+h2, loc='lower center', ncol=3, fontsize=9)