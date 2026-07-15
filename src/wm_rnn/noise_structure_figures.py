"""Research-quality figures for the completed structured-noise experiment."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from wm_rnn.config import load_config
from wm_rnn.noise_structure_experiment import (
    COLORS,
    CONDITIONS,
    STYLES,
    decoder_error,
    fit_ridge,
    generate_perturbations,
    phase_mask,
    require_cuda,
    topology_covariance,
)
from wm_rnn.training_utils import batch_to_tensors, fresh_model, generate_batch_for_task, task_config_from_dict

LABELS = {
    "unperturbed": "Unperturbed",
    "independent_gaussian": "Independent Gaussian",
    "temporally_correlated": "Temporal AR(1)",
    "context_topology_correlated": "Topology-correlated AR(1)",
}
NOISE_CONDITIONS = CONDITIONS[1:]
SEEDS = (20260714, 20260715, 20260716, 20260717, 20260718)
RMS_AXIS_LABEL = "Perturbation strength\n(realized RMS per hidden-state update)"
RMS_NOTE = "RMS is the root-mean-square perturbation amplitude, matched across active delay steps, trials, and hidden units."


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def save_figure(fig: plt.Figure, root: Path, name: str) -> None:
    fig.savefig(root / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(root / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def seed_summaries(rows: list[dict[str, str]], metric: str) -> dict[tuple[int, str, float, int], float]:
    grouped: dict[tuple[int, str, float, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["delay"]), row["condition"], float(row["strength"]), int(row["seed"]))].append(float(row[metric]))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def bootstrap_ci(values: np.ndarray, seed: int = 20260714, draws: int = 5000) -> tuple[float, float]:
    if len(values) < 2:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def mean_ci(summary: dict, delay: int, condition: str, strength: float) -> tuple[float, float, float, np.ndarray]:
    values = np.array([summary[(delay, condition, strength, seed)] for seed in SEEDS])
    low, high = bootstrap_ci(values, seed=delay + int(strength * 10000))
    return float(values.mean()), low, high, values


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=.6)


def encoding_handles() -> list[Any]:
    return [
        Line2D([0],[0],color="0.35",linewidth=.8,alpha=.45,label="Individual model seed"),
        Line2D([0],[0],color="black",linewidth=2.2,label="Mean across 5 seeds"),
        Patch(facecolor="0.5",alpha=.18,edgecolor="none",label="95% bootstrap CI across seeds"),
    ]


def plot_performance_comparison(rows: list[dict[str, str]], figure_dir: Path, data_dir: Path) -> None:
    metrics = (("response_error_degrees", "Response angular error (°)"), ("delay_decoder_error_degrees", "Delay decoder error (°)"))
    delays = (20, 80, 160); strengths = (0., .01, .025, .05, .1)
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.2), sharex=True, constrained_layout=True)
    plotted = []
    for row_idx, (metric, ylabel) in enumerate(metrics):
        summary = seed_summaries(rows, metric)
        for col, delay in enumerate(delays):
            ax = axes[row_idx, col]
            for condition in CONDITIONS:
                seed_matrix = np.array([[summary[(delay, condition, strength, seed)] for strength in strengths] for seed in SEEDS])
                for seed_values in seed_matrix:
                    ax.plot(strengths, seed_values, color=COLORS[condition], alpha=.16, linewidth=.7, linestyle=STYLES[condition])
                means=[]; lows=[]; highs=[]
                for strength in strengths:
                    mean, low, high, seed_values = mean_ci(summary, delay, condition, strength)
                    means.append(mean); lows.append(low); highs.append(high)
                    for seed, value in zip(SEEDS, seed_values):
                        plotted.append({"metric":metric,"delay":delay,"condition":condition,"strength":strength,"seed":seed,"seed_mean":value,"grand_mean":mean,"ci_low":low,"ci_high":high})
                ax.plot(strengths, means, color=COLORS[condition], linewidth=2.2, linestyle=STYLES[condition], label=LABELS[condition])
                ax.fill_between(strengths, lows, highs, color=COLORS[condition], alpha=.12, linewidth=0)
            if row_idx == 0: ax.set_title(f"Delay {delay}")
            if col == 0: ax.set_ylabel(ylabel)
            if row_idx == 1: ax.set_xlabel(RMS_AXIS_LABEL)
            style_axis(ax)
    axes[0, 2].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1, 2].legend(handles=encoding_handles(),frameon=False,fontsize=7.5,loc="upper left")
    fig.suptitle("Task performance and maintained-memory precision across five model seeds", fontsize=12)
    fig.text(.5,-.025,RMS_NOTE,ha="center",va="top",fontsize=8,color="0.3")
    save_figure(fig, figure_dir, "figure_01_performance_comparison")
    write_csv(data_dir / "figure_01_performance_comparison.csv", plotted)


def plot_condition_profiles(rows: list[dict[str, str]], figure_dir: Path, data_dir: Path) -> None:
    metrics = (
        ("response_error_degrees", "Response error (°)"),
        ("delay_decoder_error_degrees", "Decoder error (°)"),
        ("memory_drift_degrees", "Decoded drift (°)"),
        ("fixation_accuracy", "Fixation accuracy"),
    )
    delays=(20,80,160); strengths=(0.,.01,.025,.05,.1); delay_colors={20:"#009E73",80:"#0072B2",160:"#D55E00"}
    summaries={metric:seed_summaries(rows,metric) for metric,_ in metrics}
    for condition in NOISE_CONDITIONS:
        fig,axes=plt.subplots(2,2,figsize=(7.2,6.2),sharex=True,constrained_layout=True); plotted=[]
        for ax,(metric,ylabel) in zip(axes.flat,metrics):
            summary=summaries[metric]
            for delay in delays:
                means=[]; lows=[]; highs=[]
                seed_matrix=np.array([[summary[(delay,condition,strength,seed)] for strength in strengths] for seed in SEEDS])
                for seed_values in seed_matrix:
                    ax.plot(strengths,seed_values,color=delay_colors[delay],linewidth=.7,alpha=.22)
                for strength in strengths:
                    mean,low,high,seed_values=mean_ci(summary,delay,condition,strength); means.append(mean); lows.append(low); highs.append(high)
                    for seed,value in zip(SEEDS,seed_values): plotted.append({"condition":condition,"metric":metric,"delay":delay,"strength":strength,"seed":seed,"seed_mean":value,"grand_mean":mean,"ci_low":low,"ci_high":high})
                ax.plot(strengths,means,color=delay_colors[delay],linewidth=2,label=f"Delay {delay}")
                ax.fill_between(strengths,lows,highs,color=delay_colors[delay],alpha=.12,linewidth=0)
            condition_values=[float(r[metric]) for r in rows if r["condition"]==condition]
            ax.set_ylabel(ylabel); ax.set_xlabel(RMS_AXIS_LABEL if ax in axes[1,:] else ""); ax.set_ylim(max(0,min(condition_values)*.97),max(condition_values)*1.05); style_axis(ax)
        delay_handles=[Line2D([0],[0],color=delay_colors[d],linewidth=2,label=f"Delay {d}") for d in delays]
        fig.legend(handles=delay_handles+encoding_handles(),frameon=False,fontsize=7.5,ncol=3,loc="lower center",bbox_to_anchor=(.5,-.075))
        fig.suptitle(f"{LABELS[condition]}: dose and delay profile",fontsize=12)
        fig.text(.5,-.115,RMS_NOTE,ha="center",va="top",fontsize=8,color="0.3")
        stem={"independent_gaussian":"figure_02a_independent_gaussian","temporally_correlated":"figure_02b_temporal_ar1","context_topology_correlated":"figure_02c_topology_ar1"}[condition]
        save_figure(fig,figure_dir,stem); write_csv(data_dir/f"{stem}.csv",plotted)


def _autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    x=x-x.mean(); denom=np.dot(x,x)
    return np.array([np.dot(x[:len(x)-lag],x[lag:])/denom for lag in range(max_lag+1)])


def plot_manipulation_validation(config: dict[str, Any], figure_dir: Path, data_dir: Path) -> None:
    device=require_cuda(); local={**config,"task":dict(config["task"])}; local["task"]["seed"]=SEEDS[0]
    model=fresh_model(local,device); checkpoint=Path(config["paths"]["output_dir"])/"seed_sweep"/f"seed_{SEEDS[0]}"/"checkpoints"/f"yang_fixation_circular_working_memory_seed_{SEEDS[0]}.pt"; model.load_state_dict(torch.load(checkpoint,map_location=device)["model_state"]); model.eval()
    task=replace(task_config_from_dict(local,batch_size=256),delay_steps=80,seed=99117); batch=generate_batch_for_task(task); mask=phase_mask(task.seq_len,batch.phase_index,"delay",device)
    topology_target=topology_covariance(model.rnn.h2h.weight.detach()).cpu().numpy(); _,vectors=np.linalg.eigh(topology_target); unit_order=np.argsort(vectors[:,-1])
    fig,axes=plt.subplots(3,3,figsize=(9.2,7.2),sharey="row",constrained_layout=True); plotted=[]
    for col,condition in enumerate(NOISE_CONDITIONS):
        generator=torch.Generator(device=device).manual_seed(113+col)
        noise=generate_perturbations(condition,(task.seq_len,256,model.config.hidden_size),.05,mask,generator,model.rnn.h2h.weight)
        active=noise[mask].detach().cpu().numpy(); trace=active[:,0,0]; denom=np.mean(active**2); max_lag=min(20,len(trace)-1); acf=np.array([1.0]+[np.mean(active[:-lag]*active[lag:])/denom for lag in range(1,max_lag+1)]); expected=np.array([1.0]+([0.0]*max_lag if condition=="independent_gaussian" else [.9**lag for lag in range(1,max_lag+1)])); covariance=np.cov(active.reshape(-1,active.shape[-1]),rowvar=False); corr=np.corrcoef(active.reshape(-1,active.shape[-1]),rowvar=False); ordered=corr[np.ix_(unit_order,unit_order)]
        realized=float(np.sqrt(np.mean(active**2))); axes[0,col].plot(np.arange(len(trace)),trace,color=COLORS[condition],linewidth=1); axes[0,col].axhline(0,color="0.7",linewidth=.6); axes[0,col].set_title(LABELS[condition]); axes[0,col].set_ylabel("Perturbation added\nto one hidden unit" if col==0 else ""); axes[0,col].set_xlabel("Delay step"); axes[0,col].text(.03,.95,f"Realized RMS = {realized:.3f}",transform=axes[0,col].transAxes,va="top",fontsize=7.5)
        axes[1,col].plot(np.arange(len(acf)),acf,color=COLORS[condition],linewidth=1.8,label="Measured"); axes[1,col].plot(np.arange(len(expected)),expected,color="0.25",linewidth=1,linestyle=":",label="Expected"); axes[1,col].axhline(0,color="0.6",linewidth=.5); axes[1,col].set_ylim(-.15,1.05); axes[1,col].set_xlabel("Temporal lag (steps)"); axes[1,col].set_ylabel("Lag autocorrelation" if col==0 else "")
        im=axes[2,col].imshow(ordered,vmin=-1,vmax=1,cmap="coolwarm",interpolation="nearest"); axes[2,col].set_xlabel("Hidden units (topology-ordered)"); axes[2,col].set_ylabel("Hidden units\n(topology-ordered)" if col==0 else ""); offdiag=float(np.mean(np.abs(corr[~np.eye(corr.shape[0],dtype=bool)]))); axes[2,col].text(.03,.97,f"Mean |off-diagonal r| = {offdiag:.3f}",transform=axes[2,col].transAxes,va="top",fontsize=7.2,bbox={"facecolor":"white","alpha":.75,"edgecolor":"none","pad":2})
        plotted.append({"condition":condition,"realized_rms":float(np.sqrt(np.mean(active**2))),"lag1":float(acf[1]),"mean_abs_offdiag_correlation":float(np.mean(np.abs(corr[~np.eye(corr.shape[0],dtype=bool)]))),"covariance_trace":float(np.trace(covariance))})
    axes[1,2].legend(frameon=False,fontsize=7.5,loc="upper right"); fig.colorbar(im,ax=axes[2,:],label="Unit correlation",fraction=.025); fig.suptitle("Equal-strength perturbations differ in temporal persistence and population covariance",fontsize=12)
    save_figure(fig,figure_dir,"figure_03_manipulation_validation"); write_csv(data_dir/"figure_03_manipulation_validation.csv",plotted)


def _align_centroids(clean_centroids: np.ndarray, centroids: np.ndarray, angles: np.ndarray) -> tuple[np.ndarray,np.ndarray,float]:
    ideal=np.column_stack((np.cos(angles),np.sin(angles))); clean_centered=clean_centroids-clean_centroids.mean(0); u,_,vt=np.linalg.svd(clean_centered.T@ideal); rotation=u@vt; radius=np.sqrt(np.mean(np.sum((clean_centered@rotation)**2,axis=1))); radius=max(radius,1e-8)
    return clean_centered@rotation/radius,(centroids-clean_centroids.mean(0))@rotation/radius,radius


def run_hidden_state_analysis(config: dict[str, Any], output_dir: Path, delay: int=80, strength: float=.05, trials: int=320, replicates: int=5) -> tuple[list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]]]:
    device=require_cuda(); checkpoint_root=Path(config["paths"]["output_dir"])/"seed_sweep"; time_rows=[]; centroid_rows=[]; geometry_rows=[]; n_bins=16; bin_angles=np.linspace(0,2*np.pi,n_bins,endpoint=False)
    for seed in SEEDS:
        local={**config,"task":dict(config["task"])}; local["task"]["seed"]=seed; model=fresh_model(local,device); checkpoint=checkpoint_root/f"seed_{seed}"/"checkpoints"/f"yang_fixation_circular_working_memory_seed_{seed}.pt"; model.load_state_dict(torch.load(checkpoint,map_location=device)["model_state"]); model.eval()
        decoder_task=replace(task_config_from_dict(local,batch_size=512),delay_steps=delay,seed=seed+900000); decoder_batch=generate_batch_for_task(decoder_task); decoder_inputs,_,_=batch_to_tensors(decoder_batch,device)
        with torch.no_grad(): _,decoder_hidden=model(decoder_inputs)
        decoder=fit_ridge(decoder_hidden[decoder_batch.phase_index["delay"]],decoder_batch.angles,float(config.get("decoder",{}).get("ridge_alpha",1.0)))
        eval_task=replace(decoder_task,batch_size=trials,seed=seed+700000); batch=generate_batch_for_task(eval_task); inputs,_,_=batch_to_tensors(batch,device); mask=phase_mask(inputs.size(0),batch.phase_index,"delay",device)
        stored={condition:[] for condition in CONDITIONS}
        for rep in range(replicates):
            for cidx,condition in enumerate(CONDITIONS):
                generator=torch.Generator(device=device).manual_seed(seed+rep*101+cidx*10007)
                noise=generate_perturbations(condition,(inputs.size(0),trials,model.config.hidden_size),strength,mask,generator,model.rnn.h2h.weight)
                with torch.no_grad(): _,hidden=model(inputs,perturbations=noise)
                stored[condition].append(hidden.detach().cpu().numpy())
                errors=decoder_error(hidden,decoder,batch.angles).mean(1).detach().cpu().numpy(); speed=np.r_[0.0,(hidden[1:]-hidden[:-1]).norm(dim=-1).mean(1).detach().cpu().numpy()]
                for time_idx,(error,speed_value) in enumerate(zip(errors,speed)):
                    time_rows.append({"seed":seed,"replicate":rep,"condition":condition,"strength":strength,"delay":delay,"time":time_idx,"decoder_error_degrees":float(error),"hidden_speed":float(speed_value)})
        clean_all=np.concatenate(stored["unperturbed"],axis=1); delay_slice=batch.phase_index["delay"]; x=clean_all[delay_slice].reshape(-1,model.config.hidden_size); mean=x.mean(0); _,_,vt=np.linalg.svd(x-mean,full_matrices=False); components=vt[:2]
        bins=np.floor((batch.angles%(2*np.pi))/(2*np.pi)*n_bins).astype(int)
        projected={}
        raw_final={}
        for condition in CONDITIONS:
            arr=np.stack(stored[condition]) # rep,time,trial,unit
            final=arr[:,delay_slice.stop-1]; raw_final[condition]=final
            pc=(final-mean)@components.T; projected[condition]=pc
        clean_centroids=np.array([projected["unperturbed"][:,:,][...,][ :,bins==b,:].reshape(-1,2).mean(0) for b in range(n_bins)])
        for condition in CONDITIONS:
            condition_centroids=np.array([projected[condition][:,bins==b,:].reshape(-1,2).mean(0) for b in range(n_bins)])
            aligned_clean,aligned_condition,_=_align_centroids(clean_centroids,condition_centroids,bin_angles)
            for b,angle in enumerate(bin_angles): centroid_rows.append({"seed":seed,"condition":condition,"strength":strength,"delay":delay,"angle_bin":b,"angle_degrees":float(np.degrees(angle)),"clean_pc1":float(aligned_clean[b,0]),"clean_pc2":float(aligned_clean[b,1]),"condition_pc1":float(aligned_condition[b,0]),"condition_pc2":float(aligned_condition[b,1])})
            for rep in range(replicates):
                centroids=np.array([raw_final[condition][rep,bins==b].mean(0) for b in range(n_bins)])
                within=np.mean([np.linalg.norm(raw_final[condition][rep,bins==b]-centroids[b],axis=1).mean() for b in range(n_bins)])
                pairwise=np.linalg.norm(centroids[:,None,:]-centroids[None,:,:],axis=-1); between=pairwise[np.triu_indices(n_bins,1)].mean()
                clean_c=np.array([raw_final["unperturbed"][rep,bins==b].mean(0) for b in range(n_bins)]); clean_pair=np.linalg.norm(clean_c[:,None,:]-clean_c[None,:,:],axis=-1); clean_between=clean_pair[np.triu_indices(n_bins,1)].mean()
                geometry_rows.append({"seed":seed,"replicate":rep,"condition":condition,"strength":strength,"delay":delay,"within_angle_dispersion":float(within),"between_angle_separation":float(between),"separation_normalized_to_clean":float(between/clean_between)})
    write_csv(output_dir/"hidden_timecourse.csv",time_rows); write_csv(output_dir/"hidden_centroids.csv",centroid_rows); write_csv(output_dir/"hidden_geometry.csv",geometry_rows)
    return time_rows,centroid_rows,geometry_rows


def plot_timecourse(time_rows: list[dict[str,Any]], figure_dir: Path) -> None:
    grouped=defaultdict(list)
    for r in time_rows: grouped[(r["condition"],r["seed"],r["time"],"error")].append(r["decoder_error_degrees"]); grouped[(r["condition"],r["seed"],r["time"],"speed")].append(r["hidden_speed"])
    times=sorted({r["time"] for r in time_rows}); fig,axes=plt.subplots(2,1,figsize=(9,5.7),sharex=True,constrained_layout=True)
    for ax,metric,ylabel in ((axes[0],"error","Decoder error (°)"),(axes[1],"speed","Hidden-state speed")):
        for condition in CONDITIONS:
            matrix=np.array([[np.nanmean(grouped[(condition,seed,t,metric)]) for t in times] for seed in SEEDS]); mean=np.nanmean(matrix,axis=0); sem=np.nanstd(matrix,axis=0,ddof=1)/np.sqrt(len(SEEDS)); ax.plot(times,mean,color=COLORS[condition],linestyle=STYLES[condition],linewidth=2,label=LABELS[condition]); ax.fill_between(times,mean-1.96*sem,mean+1.96*sem,color=COLORS[condition],alpha=.12,linewidth=0)
        ax.set_ylabel(ylabel); style_axis(ax)
    for ax in axes:
        ax.axvspan(0,24,color="0.92",zorder=-2); ax.axvspan(25,44,color="#E5F5E0",alpha=.55,zorder=-2); ax.axvspan(45,124,color="#FFF2CC",alpha=.45,zorder=-2); ax.axvspan(125,149,color="#E8EAF6",alpha=.55,zorder=-2)
        ax.set_xlim(40,149)
    axes[0].set_ylim(0,45); axes[1].set_ylim(0,.7); axes[0].text(44,43,"Cue",ha="right",va="top",fontsize=8); axes[0].text(85,43,"Delay",ha="center",va="top",fontsize=8); axes[0].text(137,43,"Response",ha="center",va="top",fontsize=8)
    axes[0].legend(frameon=False,ncol=2,fontsize=8); axes[1].set_xlabel("Trial time step"); fig.suptitle("Memory decoding and hidden-state motion after cue presentation (delay 80, RMS 0.05)",fontsize=12)
    save_figure(fig,figure_dir,"figure_04_hidden_state_timecourse")


def _confidence_ellipse(points: np.ndarray, ax: plt.Axes, color: str) -> None:
    if len(points)<2:return
    cov=np.cov(points.T); vals,vecs=np.linalg.eigh(cov); order=vals.argsort()[::-1]; vals=vals[order]; vecs=vecs[:,order]; angle=np.degrees(np.arctan2(vecs[1,0],vecs[0,0])); width,height=2*1.96*np.sqrt(np.maximum(vals,0)/len(points)); ax.add_patch(Ellipse(points.mean(0),width,height,angle=angle,facecolor=color,edgecolor=color,alpha=.15,linewidth=1))


def plot_hidden_geometry(centroid_rows: list[dict[str,Any]], geometry_rows: list[dict[str,Any]], figure_dir: Path) -> None:
    fig=plt.figure(figsize=(10.5,7.2),constrained_layout=True); gs=fig.add_gridspec(2,4,height_ratios=[1.25,1]); top=[fig.add_subplot(gs[0,i]) for i in range(4)]; bottom=[fig.add_subplot(gs[1,:2]),fig.add_subplot(gs[1,2:])]
    for ax,condition in zip(top,CONDITIONS):
        condition_means=[]; clean_means=[]
        for b in range(16):
            pts=np.array([[r["condition_pc1"],r["condition_pc2"]] for r in centroid_rows if r["condition"]==condition and r["angle_bin"]==b]); clean=np.array([[r["clean_pc1"],r["clean_pc2"]] for r in centroid_rows if r["condition"]==condition and r["angle_bin"]==b]); p=pts.mean(0); c=clean.mean(0); _confidence_ellipse(pts,ax,COLORS[condition]); ax.arrow(c[0],c[1],p[0]-c[0],p[1]-c[1],color=COLORS[condition],alpha=.7,width=.003,head_width=.04,length_includes_head=True); ax.scatter(*p,color=COLORS[condition],s=18); ax.scatter(*c,color="0.3",s=8,alpha=.5)
            condition_means.append(p); clean_means.append(c)
        condition_means=np.array(condition_means); clean_means=np.array(clean_means); ax.plot(*np.vstack((condition_means,condition_means[0])).T,color=COLORS[condition],linewidth=.8,alpha=.55); ax.plot(*np.vstack((clean_means,clean_means[0])).T,color="0.35",linewidth=.6,alpha=.35); ax.scatter(*condition_means[0],marker="^",s=34,color=COLORS[condition],zorder=5)
        ax.set_title(LABELS[condition],fontsize=9); ax.set_aspect("equal"); ax.set_xlim(-1.65,1.65); ax.set_ylim(-1.65,1.65); ax.set_xlabel("Aligned clean-PC1\n(normalized ring radius)",fontsize=7); ax.set_ylabel("Aligned clean-PC2\n(normalized ring radius)" if condition=="unperturbed" else "",fontsize=7); ax.set_xticks([]); ax.set_yticks([])
    metrics=(("within_angle_dispersion","Within-angle dispersion"),("separation_normalized_to_clean","Between-angle separation / clean"))
    for ax,(metric,ylabel) in zip(bottom,metrics):
        positions=np.arange(len(CONDITIONS));
        for idx,condition in enumerate(CONDITIONS):
            seed_values=[]
            for seed in SEEDS:
                values=[r[metric] for r in geometry_rows if r["condition"]==condition and r["seed"]==seed]; seed_values.append(np.mean(values))
            jitter=np.linspace(-.07,.07,len(SEEDS)); ax.scatter(np.full(5,idx)+jitter,seed_values,color=COLORS[condition],s=24,alpha=.65); mean=np.mean(seed_values); low,high=bootstrap_ci(np.array(seed_values),seed=idx+72); ax.errorbar(idx,mean,yerr=[[mean-low],[high-mean]],fmt="o",color="black",capsize=3,zorder=4)
        ax.set_xticks(positions,["Clean","Independent","Temporal","Topology"],rotation=15); ax.set_ylabel(ylabel); style_axis(ax)
    geometry_handles=[Line2D([0],[0],marker="o",color="0.35",linewidth=.7,markersize=4,label="Clean angle centroid"),Line2D([0],[0],marker="o",color=COLORS["context_topology_correlated"],linewidth=.8,markersize=5,label="Perturbed angle centroid"),Line2D([0],[0],color="0.4",linewidth=1,label="Clean → perturbed displacement"),Patch(facecolor=COLORS["context_topology_correlated"],alpha=.18,label="95% CI across 5 seed centroids")]
    fig.legend(handles=geometry_handles,frameon=False,ncol=4,fontsize=7.5,loc="upper center",bbox_to_anchor=(.5,.92)); fig.suptitle("Angle-coded hidden-state geometry at the end of the delay (strength RMS = 0.05)",fontsize=12); save_figure(fig,figure_dir,"figure_05_hidden_state_geometry")


def plot_epoch_sensitivity(rows: list[dict[str,str]], figure_dir: Path, data_dir: Path) -> None:
    selected=[r for r in rows if r["condition"]=="context_topology_correlated"]; metrics=(("response_error_degrees","Response error (°)"),("delay_decoder_error_degrees","Decoder error (°)"),("fixation_accuracy","Fixation accuracy")); strengths=(0.,.01,.025,.05,.1); phases=("cue","delay","response"); colors={"cue":"#009E73","delay":"#D55E00","response":"#0072B2"}; plotted=[]
    fig,axes=plt.subplots(1,3,figsize=(10,3.2),constrained_layout=True)
    for ax,(metric,ylabel) in zip(axes,metrics):
        grouped=defaultdict(list)
        for r in selected: grouped[(r["phase"],float(r["strength"]),int(r["seed"]))].append(float(r[metric]))
        for phase in phases:
            means=[]; lows=[]; highs=[]
            for strength in strengths:
                seed_values=np.array([np.mean(grouped[(phase,strength,seed)]) for seed in SEEDS]); mean=seed_values.mean(); low,high=bootstrap_ci(seed_values,seed=int(strength*10000)+len(phase)); means.append(mean); lows.append(low); highs.append(high)
                for seed,value in zip(SEEDS,seed_values): plotted.append({"metric":metric,"phase":phase,"strength":strength,"seed":seed,"seed_mean":value,"grand_mean":mean,"ci_low":low,"ci_high":high})
            ax.plot(strengths,means,color=colors[phase],linewidth=2,label=phase.title()); ax.fill_between(strengths,lows,highs,color=colors[phase],alpha=.14,linewidth=0)
        ax.set_xlabel(RMS_AXIS_LABEL); ax.set_ylabel(ylabel); style_axis(ax)
    phase_handles=[Line2D([0],[0],color=colors[p],linewidth=2,label=p.title()) for p in phases]; axes[0].legend(handles=phase_handles,frameon=False,loc="upper left"); axes[1].legend(handles=encoding_handles()[1:],frameon=False,fontsize=7.5,loc="upper left"); fig.suptitle("Topology-correlated perturbations are most disruptive during memory maintenance",fontsize=12); fig.text(.5,-.08,RMS_NOTE,ha="center",va="top",fontsize=8,color="0.3"); save_figure(fig,figure_dir,"figure_06_epoch_sensitivity"); write_csv(data_dir/"figure_06_epoch_sensitivity.csv",plotted)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/yang_fixation_circular_working_memory.yaml"); parser.add_argument("--root",type=Path,default=Path("outputs/yang_fixation_circular_working_memory/perturbation_experiments/noise_structure")); parser.add_argument("--reuse-hidden",action="store_true"); args=parser.parse_args(); config=load_config(args.config); figure_dir=args.root/"figures"; data_dir=args.root/"figure_data"; figure_dir.mkdir(parents=True,exist_ok=True); data_dir.mkdir(parents=True,exist_ok=True)
    main_rows=read_csv(args.root/"full"/"summary_metrics.csv"); timing_rows=read_csv(args.root/"epoch_timing_full"/"summary_metrics.csv")
    plot_performance_comparison(main_rows,figure_dir,data_dir); plot_condition_profiles(main_rows,figure_dir,data_dir); plot_manipulation_validation(config,figure_dir,data_dir)
    if args.reuse_hidden and (data_dir/"hidden_timecourse.csv").exists():
        time_rows=[{**r,"seed":int(r["seed"]),"replicate":int(r["replicate"]),"time":int(r["time"]),"decoder_error_degrees":float(r["decoder_error_degrees"]),"hidden_speed":float(r["hidden_speed"])} for r in read_csv(data_dir/"hidden_timecourse.csv")]; centroid_rows=[{**r,"seed":int(r["seed"]),"angle_bin":int(r["angle_bin"]),"condition_pc1":float(r["condition_pc1"]),"condition_pc2":float(r["condition_pc2"]),"clean_pc1":float(r["clean_pc1"]),"clean_pc2":float(r["clean_pc2"])} for r in read_csv(data_dir/"hidden_centroids.csv")]; geometry_rows=[{**r,"seed":int(r["seed"]),"replicate":int(r["replicate"]),"within_angle_dispersion":float(r["within_angle_dispersion"]),"separation_normalized_to_clean":float(r["separation_normalized_to_clean"])} for r in read_csv(data_dir/"hidden_geometry.csv")]
    else: time_rows,centroid_rows,geometry_rows=run_hidden_state_analysis(config,data_dir)
    plot_timecourse(time_rows,figure_dir); plot_hidden_geometry(centroid_rows,geometry_rows,figure_dir); plot_epoch_sensitivity(timing_rows,figure_dir,data_dir)


if __name__ == "__main__": main()
