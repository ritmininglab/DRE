import numpy as np 
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, average_precision_score
from matplotlib import pyplot 
import matplotlib.pyplot as plt 
from matplotlib import pyplot 
import matplotlib.patches as mpatches
from itertools import combinations

def draw_reliability_graph(ece, bin_acc, bins, prune, arch, loss_type):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    ax.set_axisbelow(True)
    ax.grid(color = 'gray', linestyle = 'dashed')
    plt.bar(bins, bins, width = 0.1, alpha = 0.3, edgecolor = 'black', color = 'r', hatch = '\\')
    plt.bar(bins, bin_acc, width = 0.1, alpha = 0.1, edgecolor = 'black', color = 'b')
    plt.plot([0, 1], [0, 1], '--', color = 'gray', linewidth = 2)
    plt.gca().set_aspect('equal', adjustable = 'box')
    ECE_patch = mpatches.Patch(color = 'green', label = 'ECE: {:.2f}%'.format(ece*100))
    plt.legend(handles = [ECE_patch])
    plt.savefig('hists/ece_plot_'+arch+'_'+spar+'_'+loss_type+'.png', bbox_inches = 'tight')
    plt.clf()

def draw_histogram(conf, gt, pred, prune, arch, n_bins = 10):
    no_correct, no_incorrect = np.zeros(n_bins), np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m/n_bins, (m+1)/n_bins
        for i in range(len(gt)):
            if conf[i]>a and conf[i]<=b:
                if gt[i]==pred[i]:
                    no_correct[m]+=1
                else:
                    no_incorrect[m]+=1

    width = 0.55
    confs = ['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
    ind = np.arange(n_bins)
    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(ind, no_correct, width, color = 'r')
    ax.bar(ind, no_incorrect, width, bottom = no_correct, color = 'b')
    ax.set_ylabel('Number of Samples', fontsize = 20)
    ax.set_xlabel('Confidence', fontsize = 20)
    ax.set_xticks(ind, confs, fontsize = 20, rotation = 90)
    ax.legend(labels = ['Correct', 'Incorrect'])
    plt.savefig('hists/correct_incorrect_'+arch+'_'+prune+'.png', bbox_inches = 'tight')
    plt.clf()


def ece_score(gt, pred, conf, n_bins = 10):
    bin_acc, bin_conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m/n_bins, (m+1)/n_bins
        for i in range(len(gt)):
            if conf[i]>a and conf[i]<=b:
                Bm[m]+=1
                if gt[i]==pred[i]:
                    bin_acc[m]+=1
                bin_conf[m]+=conf[i]
        if Bm[m]!=0:
            bin_acc[m] = bin_acc[m]/Bm[m]
            bin_conf[m] = bin_conf[m]/Bm[m]
    
    ece = 0
    acc = 0
    for m in range(n_bins):
        ece+=Bm[m]*np.abs((bin_acc[m]-bin_conf[m]))
        acc+=Bm[m]*bin_acc[m]

    return ece/sum(Bm), acc/sum(Bm), bin_acc, Bm


if __name__=="__main__":
    archs = ['cResNet101']
    n_bins = 10
    bins = np.linspace(0, 1, 10)
    spar = '0.05'

    for arch in archs:
        print("\n Working on an architecture", arch)
        bins = np.linspace(0, 1, 10)
        run = 2
        all_lambs = [1, 5, 10, 15, 20, 25, 30, 50, 500]
        for  i in range(len(all_lambs)):
            for j in range(i, len(all_lambs)):
                a = all_lambs[i]
                b = all_lambs[j]

                run_preds = []
                run_gts = []
                run_outputs = []

                lambs = ['',  a,  b]
                for lamb in lambs:

                    if lamb =='':
                        identifier = arch+'_CIFAR10_'+spar+'_'+str(2)+'.npy'
                        
                    elif lamb==500:
                        
                        identifier = arch+'_CIFAR10_'+spar+'_'+str(1)+'.npy'
                    else:
                        identifier = arch+'_CIFAR10_'+spar+'_'+str(lamb)+'_'+str(run)+'.npy'
                        

                    in_outputs = np.load("outputs/output_"+identifier)
                    run_outputs.append(in_outputs)
                    gts = np.load('outputs/gts_'+identifier)
                    run_pred = in_outputs.argmax(axis = 1)
                    in_outputs = softmax(in_outputs, axis = 1)
                    ece, acc, bin_acc, _ = ece_score(gts, in_outputs.argmax(axis = 1), in_outputs.max(axis = 1), n_bins = len(bins))
                    print("Accuracy for a run", run, acc, "ece", ece)
                    run_preds.append(in_outputs)
                    run_gts.append(gts)

                run_outputs = np.array(run_outputs)
                total_outputs = np.mean(run_outputs, axis = 0)
                softmax_total_outputs = softmax(total_outputs, axis = 1)
                total_pred = softmax_total_outputs.argmax(axis =1)
                total_conf = softmax_total_outputs.max(axis = 1)
                gt = np.mean(run_gts, axis = 0)

                ece, acc, bin_acc, _ = ece_score(gt, total_pred, total_conf, n_bins = len(bins))
                print("For combinarion", "(", a, b, ")"  , "sparsity:", spar, "accuracy:", acc, "ece", ece)


            