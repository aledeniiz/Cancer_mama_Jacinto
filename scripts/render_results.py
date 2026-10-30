"""Re-render archived aggregate evidence; never fit models or read patient records."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

def main():
    study=json.loads((ROOT/'results/study.json').read_text(encoding='utf-8'))
    out=ROOT/'results/figures';out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
        'svg.fonttype':'none','svg.hashsalt':'breastdce-aggregate-v1','axes.titleweight':'bold'})
    colors=['#849aa9','#007c8a','#5b4fbb','#be7640']
    def save(fig,name):
        fig.savefig(out/name,format='svg',bbox_inches='tight',metadata={'Date':None,
            'Description':'Aggregate archived evidence; no patient data or score thresholds.'})
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,5.2))
    for m,color in zip(study['oof']['models'],colors):
        ax.plot([p['fpr'] for p in m['roc']],[p['tpr'] for p in m['roc']],color=color,
            linewidth=1.8,label=f"{m['label']} ({m['auc']:.4f})")
    ax.plot([0,1],[0,1],ls='--',color='#7d8991',lw=1)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='False positive rate',ylabel='True positive rate',
        title='Matched OOF · 1,092 patients · five folds')
    ax.legend(loc='lower right',fontsize=9,frameon=False);ax.grid(alpha=.15)
    save(fig,'oof_roc.svg')
    fig,ax=plt.subplots(figsize=(8.4,4.2))
    ms=study['oof']['models'];ax.barh(range(4),[m['auc'] for m in ms],color=colors,height=.58)
    ax.set_yticks(range(4),[m['label'] for m in ms]);ax.invert_yaxis()
    for i,m in enumerate(ms):ax.text(m['auc']+.015,i,f"{m['auc']:.4f}",va='center')
    ax.axvline(.5,ls='--',lw=1,color='#7d8991');ax.set(xlim=(0,1),xlabel='ROC-AUC',
        title='Clinical-only has the highest observed OOF AUC')
    ax.grid(axis='x',alpha=.15);save(fig,'oof_comparison.svg')
    for key,name,title in [('oof','fusion_bce.svg','Neural fusion · mean of five OOF fold curves'),
        ('historical_development','historical_bce.svg','Historical EMA · reused development holdout')]:
        rows=study[key]['losses_mean'] if key=='oof' else study[key]['history']
        fig,ax=plt.subplots(figsize=(7.2,4.4))
        ax.plot([r['epoch'] for r in rows],[r['train_bce'] for r in rows],label='Train',color='#007c8a',lw=2)
        ax.plot([r['epoch'] for r in rows],[r['validation_bce'] for r in rows],label='Validation',color='#5b4fbb',lw=2)
        ax.set(xlim=(1,rows[-1]['epoch']),ylim=(0,max(.8,max(r['validation_bce'] for r in rows)*1.05)),
            xlabel='Epoch',ylabel='Binary cross-entropy',title=title)
        ax.legend(frameon=False);ax.grid(alpha=.15);save(fig,name)
    print('Rendered four figures from public aggregate evidence. No training or individual inputs.')

if __name__=='__main__':main()
