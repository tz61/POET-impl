#!/usr/bin/env python3
"""Create publication-style figures and a short XeLaTeX report.

Reads validated, completed measurements only. Run with system Python -I.
"""
from pathlib import Path
import hashlib
import json
import statistics
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output'
MODES=['full','topk','no_transfer']
NAMES={'full':'Full','topk':'Top-1','no_transfer':'No transfer'}
COLORS={'full':'#297eae','topk':'#149675','no_transfer':'#8a65b9'}


def plots(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.labelcolor':'#283340','text.color':'#283340',
                         'axes.edgecolor':'#a9b3bc','xtick.color':'#536170',
                         'ytick.color':'#536170','pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),gridspec_kw={'wspace':.33})
    for k,m in enumerate(MODES):
        group=data['groups'][m]
        for ax,vals,scale in [(axes[0],group['final_returns_by_seed'],1.),(axes[1],group['training_world_steps_by_seed'],1000.)]:
            ys=[v/scale for v in vals]
            ax.scatter([k-.11,k,k+.11],ys,color=COLORS[m],s=35,zorder=3)
            ax.plot([k-.22,k+.22],[statistics.mean(ys)]*2,color=COLORS[m],lw=2.5,zorder=3)
            ax.grid(axis='y',alpha=.16)
            ax.set_xticks(range(3));ax.set_xticklabels([NAMES[x] for x in MODES])
    axes[0].set_ylabel('Mean best-of-library test return')
    axes[1].set_ylabel('Training world steps (thousands)')
    axes[0].set_title('Performance',loc='left',fontweight='bold')
    axes[1].set_title('Compute',loc='left',fontweight='bold')
    fig.suptitle(f"Matched checkpoint: {data['matched_checkpoint_iteration']} optimization rounds",x=.07,ha='left',fontsize=13,fontweight='bold')
    fig.subplots_adjust(top=.78,bottom=.19,left=.09,right=.98)
    fig.text(.09,.025,'Each dot is one training seed; short bars are means. Equal rounds do not imply equal computation.',fontsize=8,color='#657180')
    fig.savefig(OUT/'physics_comparison_figure.pdf')
    fig.savefig(OUT/'physics_comparison_figure.png',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,3.6))
    cutoff=data['matched_checkpoint_iteration']
    for m in MODES:
        for k,run in enumerate(sorted((r for r in data['runs'] if r['mode']==m),key=lambda r:r['seed'])):
            points=[p for p in run['checkpoints'] if p['iteration']<=cutoff]
            ax.plot([p['train_metrics']['simulated_world_steps']/1000 for p in points],
                    [p['evaluation']['mean_best_library_return'] for p in points],
                    color=COLORS[m],alpha=.65,marker=['o','s','^'][k],ms=4,lw=1.2,
                    label=f"{NAMES[m]}, seed {run['seed']}")
    ax.set_xlabel('Actual cumulative training world steps (thousands)')
    ax.set_ylabel('Mean best-of-library test return')
    ax.grid(alpha=.16);ax.legend(ncol=3,fontsize=8,frameon=False,loc='best')
    fig.tight_layout();fig.savefig(OUT/'physics_learning_curves.pdf');plt.close(fig)


def tex_report(data):
    matched=data['matched_checkpoint_iteration']
    groups=data['groups']; runs=data['comparison_runs']
    rows=[]
    for m in MODES:
        g=groups[m];returns=g['final_returns_by_seed'];steps=g['training_world_steps_by_seed']
        rows.append(f"{NAMES[m]} & {statistics.mean(returns):.4f} & ${min(returns):.4f}$ to ${max(returns):.4f}$ & {statistics.mean(steps):,.0f} & {sum(g['goal_tasks_by_seed'])}/12 " + chr(92)*2)
    maxx=max(r['heldout_max_x'] for r in runs)
    goals=sum(r['evaluation']['tasks_reached_goal'] for r in runs)
    gain=groups['topk']['final_mean_best_library_return']-groups['full']['final_mean_best_library_return']
    work=100*(1-statistics.mean(groups['topk']['training_world_steps_by_seed'])/statistics.mean(groups['full']['training_world_steps_by_seed']))
    terminal='; '.join(NAMES[m]+': '+', '.join(str(r['iterations_completed']) for r in
        sorted((r for r in data['runs'] if r['mode']==m),key=lambda r:r['seed'])) for m in MODES)
    env=data['environments'][0]
    totals=data['totals'];sourcehash=hashlib.sha256((ROOT/'evidence/physics_comparison.json').read_bytes()).hexdigest()
    interpretation=(r'No condition completes a test course at the common checkpoint, so we cannot claim walking was learned or that generalization across obstacles was validated.' if goals==0 else r'Some policies complete a test course, but this small fixed-task experiment still cannot show the advantage of complete open-ended environment evolution.')
    location=(r'The farthest position is still inside the initial flat section: the start is at $x=2$ m and terrain changes begin after $x=4$ m. The terrain preview input may influence actions earlier, but that does not mean the body has crossed an obstacle.' if maxx<4 else r'Some policies moved beyond the initial flat section; the farthest position alone cannot show which obstacle was crossed, so trajectories must be checked one by one.')
    body=r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=19mm,headheight=14pt]{geometry}
\usepackage{fontspec,amsmath,booktabs,graphicx,hyperref,fancyhdr,xcolor,titlesec}
\setmainfont{Lato}
\definecolor{ink}{HTML}{193B4C}\definecolor{muted}{HTML}{627483}
\hypersetup{colorlinks=true,urlcolor=ink,pdftitle={POET physics comparison, preliminary report: Team 33 / Paper 44}}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small\color{muted}CS8803 DRL / Team 33 / Paper ID 44}
\fancyhead[R]{\small\color{muted}22 October 2026}\fancyfoot[R]{\thepage}
\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}\linespread{1.14}
\titleformat{\section}{\Large\sffamily\bfseries\color{ink}}{}{0pt}{}
\begin{document}
{\Huge\sffamily\bfseries\color{ink}POET physics comparison}\par
{\large POET Top-1 transfer screen: preliminary comparison and reproducible record}\par
\textbf{Scope of this report:} actual physics training on fixed tasks, not a successful reproduction of complete open-ended POET. All figures are generated from saved per-call evaluation records.

\section{What we actually did}
We ran Isaac Lab 3 Beta 2 with Newton / Featherstone on CPU, in the requested \texttt{\string~/env\_isaaclab} environment. Each condition has four fixed training tasks and four independently initialized policies, comparing Full transfer, Top-1 (adapting only the highest-ranked donor), and No transfer. The code calls \texttt{POET.optimize} and \texttt{POET.transfer} from the production implementation directly.

The three seeds are 24582922, 24582923, and 24582924. The same seed uses identical initial policies across the three methods. Transfer is scheduled after optimization rounds 5, 10, 15, and 20. Each policy uses 4 antithetic ES perturbation pairs per optimization, with four parallel physics worlds, 60 Hz control, and at most 600 steps per episode. Tasks do not mutate or reproduce, so these results cannot serve as a comparison of complete environment evolution.

\section{What is the same and what differs}
The training tasks are flat ground, small roughness, low stumps, and narrow gaps. The fixed test set contains roughness, stumps, wider gaps, and mixed terrain, and the actual geometry hashes of all four test terrains differ from the training set. Each checkpoint evaluates all four current policies on the four test terrains, with no test-time adaptation. The metric is
\[
\bar R=\frac14\sum_{e\in\mathcal E_{test}}\max_{i=1,\ldots,4}R_e(\theta_i).
\]
It measures policy-library coverage with test-time oracle selection, not the generalization of one policy, and the 16 evaluation trajectories must not be treated as 16 independent training replicates.

Each condition is planned for up to 20 rounds with a 900-second soft time limit. To avoid inconsistent rounds from early stopping, the main comparison uses the largest checkpoint saved by all nine conditions: \textbf{round @MATCHED@}. This matches optimization rounds only; transfer changes policy evaluation counts and episode lengths, so actual compute is recorded separately. Actual terminal rounds per method, in the seed order above, are: @TERMINAL@. Original end points and the common checkpoint are both kept.

\section{Limits of the conclusions}
@INTERPRETATION@ At this common checkpoint, \textbf{@GOALS@} of 36 ``method $\times$ seed $\times$ test task'' combinations have a policy reaching the goal; the farthest test position is \textbf{@MAXX@ m}. @LOCATION@
\newpage
\section{Performance and cost at the same checkpoint}
\includegraphics[width=\linewidth]{physics_comparison_figure.pdf}
\begin{center}\small
\begin{tabular}{lrrrr}\toprule
Method & Mean test return & Seed range & Mean training world steps & Reached goal\\\midrule
@ROWS@
\bottomrule\end{tabular}\end{center}

Compared with Full, Top-1's mean test return differs by \textbf{@GAIN@}, and its change in mean training world steps corresponds to a \textbf{@WORK@\% saving} (a negative value means more was used). The three dots are three independent training runs, and the line is only their mean; the sample is small, so no significance or general superiority is claimed. World steps include completed or padded parallel lanes, and wall time for initialization and scene rebuilding is stored separately. Test cost is not mixed into the training axis.

\includegraphics[width=\linewidth]{physics_learning_curves.pdf}
\small The curves only connect actually saved checkpoints, with no smoothing or interpolation; different symbols of the same color are different seeds. The horizontal axis is actual cumulative training world steps, so equal rounds can sit at different positions.\normalsize
\newpage
\section{What this data can support}
This run validates that the actual physics training path for Top-1 runs, and lets us observe differences between return and compute. Full transfer and Top-1 both keep all direct transfer candidates; screening only affects the later adaptation proposals. No transfer provides an independent-optimization control. This implementation differs from the paper's Box2D task in dynamics, terrain, and some observations, so returns from the two cannot be drawn on one comparison scale.

The synthetic transfer experiment stays as mechanism evidence: with four tasks and 32 perturbation pairs per adaptation, Full / Top-1 use 792 / 272 policy evaluations. The 65.66\% is the evaluation saving for that one synthetic event and cannot stand in for this report's actual change in training world steps. The two-peak counterexample also shows that a donor with a lower direct score may be better after adaptation; there is a tradeoff between speed and search coverage.

\section{How to strengthen the conclusions}
First diagnose flat-ground motion: observe the initial pose, termination conditions, action scale, and policy updates, and confirm the full baseline can move forward steadily. The episode limit here is only 10 seconds, so going from $x=2$ m to $x=30$ m needs an average of at least 2.8 m/s; longer-term tasks should extend the horizon and freeze a new task fingerprint. Then increase the training budget and ES perturbation samples, enable environment mutation and reproduction, and check new-environment admission and transfer behavior. Once the full baseline can learn, repeat the comparison on an equal world-step budget, with more seeds and independent test layouts.

A short GPU test on this machine actually succeeded, but the main comparison uses the same CPU / Featherstone backend throughout; the installed version is Beta 2, not yet the target GA dependency set. These data must not be relabeled as GA or GPU training results. The course Project note does not require physical robot hardware experiments; this uses physics engine simulation.

\section{Reproduction entry points and audit}
Runner: \texttt{experiments/run\_physics\_comparison.py}. Batch directories store \texttt{protocol.json} and environment versions; each condition stores policy checkpoints, per-call evaluation JSONL, transfer event JSONL, and a summary. An independent audit recomputes counts, task fingerprints, initial policies, and the match between test policies and checkpoints. The common point is chosen from checkpoint numbers only, not from performance.

The nine selected complete runs total @TRAINSTEPS@ training world steps and @EVALSTEPS@ test world steps; this is a different scope from the main figure's common-checkpoint budget. Six earlier attempts were interrupted with exit code 143, cause undetermined; the original records are kept, and all were rerun with the original parameters, not selected by score. Evaluations unreturned at interruption cannot be counted exactly, so the totals above exclude those attempts and are not whole-project compute. The main data file is \texttt{evidence/physics\_comparison.json}, SHA-256:\par
{\tiny\ttfamily @HASH@}\par
Rebuild the figures and this report with \texttt{/usr/bin/python3 -I experiments/build\_experiment\_report.py}. The video source is in the POET-slides repository.

\section{Sources}
Wang et al. (2019), \emph{Paired Open-Ended Trailblazer (POET): Endlessly Generating Increasingly Complex and Diverse Learning Environments and Their Solutions}. \url{https://arxiv.org/abs/1901.01753v3}. Original paper results, the synthetic mechanism experiment, and our physics results are labeled separately in the video and data.
\end{document}
'''
    values={'MATCHED':str(matched),'TERMINAL':terminal,'INTERPRETATION':interpretation,
            'GOALS':str(goals),'MAXX':f'{maxx:.3f}','LOCATION':location,
            'ROWS':'\n'.join(rows),'GAIN':f'{gain:+.4f}','WORK':f'{work:.2f}',
            'TRAINSTEPS':f"{totals['train']['simulated_world_steps']:,}",
            'EVALSTEPS':f"{totals['eval']['simulated_world_steps']:,}",'HASH':sourcehash}
    for key,val in values.items():body=body.replace('@'+key+'@',val)
    tex=OUT/'POET_Team33_Simulation_Report.tex';tex.write_text(body)
    for _ in range(2):
        subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error',tex.name],cwd=OUT,
                       stdout=subprocess.DEVNULL,check=True)
    print(OUT/'POET_Team33_Simulation_Report.pdf')


if __name__=='__main__':
    data=json.loads((ROOT/'evidence/physics_comparison.json').read_text())
    if not data.get('complete_three_by_three_matrix') or data.get('primary_comparison_status')!='ready':
        raise SystemExit('Need a complete, validated 3x3 matrix and a common checkpoint first.')
    OUT.mkdir(exist_ok=True);plots(data);tex_report(data)
