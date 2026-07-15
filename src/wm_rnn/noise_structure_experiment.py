"""CUDA-only frozen-model structured-noise experiment and figure suite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from wm_rnn.config import load_config
from wm_rnn.training_utils import batch_to_tensors, fresh_model, generate_batch_for_task, task_config_from_dict
from wm_rnn.tuned_task import circular_angular_error, decode_population_angle

CONDITIONS = ("unperturbed", "independent_gaussian", "temporally_correlated", "context_topology_correlated")
COLORS = {"unperturbed": "#000000", "independent_gaussian": "#0072B2", "temporally_correlated": "#E69F00", "context_topology_correlated": "#CC79A7"}
STYLES = {"unperturbed": "-", "independent_gaussian": "--", "temporally_correlated": "-.", "context_topology_correlated": ":"}


def require_cuda() -> torch.device:
    """Return CUDA device or fail; this experiment never falls back to CPU."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for the noise-structure experiment; CPU fallback is disabled")
    return torch.device("cuda")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phase_mask(seq_len: int, phase_index: dict[str, slice], phase: str, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    selected = "fixation" if phase == "cue" and "cue" not in phase_index else phase
    slc = phase_index[selected]
    if phase == "cue" and selected == "fixation":
        slc = phase_index["cue"]
    mask[slc] = True
    return mask


def topology_covariance(recurrent_weight: torch.Tensor, mixture: float = 0.75) -> torch.Tensor:
    """Normalized covariance ``(1-m) I + m W W^T`` with unit mean variance."""
    gram = recurrent_weight @ recurrent_weight.T
    gram = gram / torch.diagonal(gram).mean().clamp_min(torch.finfo(gram.dtype).eps)
    covariance = (1.0 - mixture) * torch.eye(gram.size(0), device=gram.device) + mixture * gram
    return covariance / torch.diagonal(covariance).mean()


def generate_perturbations(
    condition: str,
    shape: tuple[int, int, int],
    strength: float,
    active_mask: torch.Tensor,
    generator: torch.Generator,
    recurrent_weight: torch.Tensor,
    rho: float = 0.9,
    topology_mixture: float = 0.75,
    topology_transform: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate CUDA perturbations and match realized RMS over active entries."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    device = recurrent_weight.device
    noise = torch.zeros(shape, device=device)
    if condition == "unperturbed" or strength == 0 or not active_mask.any():
        return noise
    raw = torch.randn(shape, device=device, generator=generator)
    if condition != "independent_gaussian":
        raw[0] = raw[0] / np.sqrt(max(1e-12, 1.0 - rho * rho))
        innovation = np.sqrt(1.0 - rho * rho)
        for time in range(1, shape[0]):
            raw[time] = rho * raw[time - 1] + innovation * raw[time]
    if condition == "context_topology_correlated":
        if topology_transform is None:
            covariance = topology_covariance(recurrent_weight.detach(), topology_mixture)
            transform = torch.linalg.cholesky(covariance + 1e-6 * torch.eye(shape[2], device=device))
        else:
            transform = topology_transform
        raw = raw @ transform.T
    noise[active_mask] = raw[active_mask]
    rms = noise[active_mask].square().mean().sqrt()
    noise[active_mask] *= float(strength) / rms.clamp_min(torch.finfo(noise.dtype).eps)
    return noise


def generate_perturbation_grid(
    variants: list[tuple[str, float]],
    shape: tuple[int, int, int],
    active_mask: torch.Tensor,
    generator: torch.Generator,
    topology_transform: torch.Tensor,
    rho: float = 0.9,
) -> list[torch.Tensor]:
    """Vectorize generation across condition/strength variants on CUDA."""
    time_steps, batch_size, hidden_size = shape
    device = topology_transform.device
    raw = torch.randn((len(variants), time_steps, batch_size, hidden_size), device=device, generator=generator)
    correlated = torch.tensor(
        [condition in {"temporally_correlated", "context_topology_correlated"} for condition, _ in variants],
        device=device,
    )
    if correlated.any():
        innovation = np.sqrt(1.0 - rho * rho)
        for time_idx in range(1, time_steps):
            raw[correlated, time_idx] = rho * raw[correlated, time_idx - 1] + innovation * raw[correlated, time_idx]
    topology = torch.tensor([condition == "context_topology_correlated" for condition, _ in variants], device=device)
    if topology.any():
        raw[topology] = raw[topology] @ topology_transform.T
    noise = torch.zeros_like(raw)
    noise[:, active_mask] = raw[:, active_mask]
    strengths = torch.tensor(
        [0.0 if condition == "unperturbed" else strength for condition, strength in variants],
        device=device,
        dtype=noise.dtype,
    )
    rms = noise[:, active_mask].square().mean(dim=(1, 2, 3)).sqrt()
    scale = strengths / rms.clamp_min(torch.finfo(noise.dtype).eps)
    noise *= scale[:, None, None, None]
    return list(noise.unbind(0))


def fit_ridge(hidden: torch.Tensor, angles: np.ndarray, alpha: float = 1.0) -> torch.Tensor:
    """Fit a sine/cosine ridge decoder on CUDA."""
    x = hidden.reshape(-1, hidden.size(-1))
    targets = torch.as_tensor(np.column_stack((np.sin(angles), np.cos(angles))), dtype=x.dtype, device=x.device)
    repeats = hidden.size(0)
    y = targets.repeat(repeats, 1)
    eye = torch.eye(x.size(1), device=x.device, dtype=x.dtype)
    return torch.linalg.solve(x.T @ x + alpha * eye, x.T @ y)


def decoder_error(hidden: torch.Tensor, decoder: torch.Tensor, angles: np.ndarray) -> torch.Tensor:
    decoded = hidden @ decoder
    predicted = torch.atan2(decoded[..., 0], decoded[..., 1])
    target = torch.as_tensor(angles, device=hidden.device)[None, :]
    error = torch.abs(torch.remainder(predicted - target + torch.pi, 2 * torch.pi) - torch.pi)
    return torch.rad2deg(error)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run_experiment(config: dict[str, Any], seeds: Iterable[int], delays: Iterable[int], strengths: Iterable[float], replicates: int, batches: int, batch_size: int, decoder_trials: int, phases: Iterable[str], output: Path, variant_chunk_size: int = 4) -> dict[str, Any]:
    device=require_cuda(); output.mkdir(parents=True,exist_ok=True); rows=[]; diagnostics=[]; hashes={}
    seeds=list(seeds); delays=list(delays); strengths=list(strengths); phases=list(phases)
    variants=[(condition,float(strength)) for condition in CONDITIONS for strength in strengths]
    total_blocks=len(seeds)*len(delays)*len(phases)*replicates*batches; completed=0; started=time.perf_counter()
    checkpoint_root=Path(config["paths"]["output_dir"])/"seed_sweep"
    for seed in seeds:
        checkpoint=checkpoint_root/f"seed_{seed}"/"checkpoints"/f"yang_fixation_circular_working_memory_seed_{seed}.pt"; hashes[str(checkpoint)]=sha256(checkpoint)
        local=json.loads(json.dumps(config)); local["task"]["seed"]=int(seed); model=fresh_model(local,device); model.load_state_dict(torch.load(checkpoint,map_location=device)["model_state"]); model.eval()
        covariance=topology_covariance(model.rnn.h2h.weight.detach()); topology_transform=torch.linalg.cholesky(covariance+1e-6*torch.eye(model.config.hidden_size,device=device))
        for delay in delays:
            task=replace(task_config_from_dict(local,batch_size=decoder_trials),delay_steps=int(delay),seed=int(seed)+900000); clean=generate_batch_for_task(task); x,_,_=batch_to_tensors(clean,device)
            with torch.no_grad(): _,hclean=model(x)
            decoder=fit_ridge(hclean[clean.phase_index["delay"]],clean.angles,float(config.get("decoder",{}).get("ridge_alpha",1.0)))
            for phase in phases:
                for rep in range(replicates):
                    for batch_idx in range(batches):
                        eval_task=replace(task,batch_size=batch_size,seed=int(seed)+100000+rep*1000+batch_idx); batch=generate_batch_for_task(eval_task); inputs,targets,_=batch_to_tensors(batch,device); mask=phase_mask(inputs.size(0),batch.phase_index,phase,device)
                        gen=torch.Generator(device=device).manual_seed(int(seed)+int(delay)*10000+rep*100+batch_idx)
                        noises=generate_perturbation_grid(variants,(inputs.size(0),batch_size,model.config.hidden_size),mask,gen,topology_transform)
                        logits_parts=[]; hidden_parts=[]
                        for chunk_start in range(0,len(variants),variant_chunk_size):
                            chunk_noises=noises[chunk_start:chunk_start+variant_chunk_size]; combined_noise=torch.cat(chunk_noises,dim=1); combined_inputs=inputs.repeat(1,len(chunk_noises),1)
                            with torch.no_grad(): chunk_logits,chunk_hidden=model(combined_inputs,perturbations=combined_noise)
                            logits_parts.append(chunk_logits); hidden_parts.append(chunk_hidden)
                        combined_logits=torch.cat(logits_parts,dim=1); combined_hidden=torch.cat(hidden_parts,dim=1)
                        response=batch.phase_index["response"]; delay_slice=batch.phase_index["delay"]
                        delay_grid=combined_hidden[delay_slice].reshape(delay_slice.stop-delay_slice.start,len(variants),batch_size,model.config.hidden_size).permute(1,0,2,3).reshape(len(variants),-1,model.config.hidden_size)
                        covariance_grid=delay_grid-delay_grid.mean(dim=1,keepdim=True); covariance_grid=covariance_grid.transpose(1,2)@covariance_grid/max(1,delay_grid.size(1)-1); eigenvalues=torch.linalg.eigvalsh(covariance_grid).clamp_min(0); participation_grid=eigenvalues.sum(dim=1).square()/eigenvalues.square().sum(dim=1).clamp_min(1e-12)
                        for variant_idx,(condition,strength) in enumerate(variants):
                            slc=slice(variant_idx*batch_size,(variant_idx+1)*batch_size); logits=combined_logits[:,slc]; hidden=combined_hidden[:,slc]; noise=noises[variant_idx]
                            pred=decode_population_angle(logits[response,...,:len(batch.preferred_angles)].detach().cpu().numpy(),batch.preferred_angles); targets_np=np.broadcast_to(batch.angles,(pred.shape[0],len(batch.angles))); response_error=float(np.degrees(circular_angular_error(pred,targets_np)).mean()); derr=decoder_error(hidden[delay_slice],decoder,batch.angles)
                            decoded=hidden[delay_slice]@decoder; decoded_angles=torch.atan2(decoded[...,0],decoded[...,1]); drift=torch.rad2deg(torch.abs(torch.remainder(decoded_angles[-1]-decoded_angles[0]+torch.pi,2*torch.pi)-torch.pi)).mean()
                            centered=hidden[delay_slice]-hidden[delay_slice].mean(dim=1,keepdim=True); dispersion=centered.square().sum(-1).mean().sqrt(); sep=hidden[delay_slice].mean(0).var(0).mean().sqrt(); pr=participation_grid[variant_idx]; binary=(hidden[delay_slice]>hidden[delay_slice].median()).flatten(); transitions=(binary[1:]!=binary[:-1]).float().mean(); fix=(logits[...,-1]>=.5)==(targets[...,-1]>=.5)
                            rows.append({"seed":seed,"delay":delay,"phase":phase,"replicate":rep,"batch":batch_idx,"condition":condition,"strength":strength,"response_error_degrees":response_error,"delay_decoder_error_degrees":float(derr.mean()),"memory_drift_degrees":float(drift),"fixation_accuracy":float(fix.float().mean()),"trajectory_speed":float((hidden[1:]-hidden[:-1]).norm(dim=-1).mean()),"within_angle_dispersion":float(dispersion),"between_angle_separation":float(sep),"participation_ratio":float(pr),"lzc_normalized":float(transitions),"recovery":float((hidden[-1]-hidden[delay_slice.stop-1]).norm(dim=-1).mean())})
                            active=noise[mask]; lag=float((active[1:]*active[:-1]).mean()/active.square().mean().clamp_min(1e-12)) if active.size(0)>1 else 0.0; diagnostics.append({"seed":seed,"delay":delay,"phase":phase,"replicate":rep,"batch":batch_idx,"condition":condition,"strength":strength,"realized_rms":float(active.square().mean().sqrt()) if active.numel() else 0.0,"lag1":lag,"outside_max":float(noise[~mask].abs().max()) if (~mask).any() else 0.0})
                        completed+=1
                        if completed % 10 == 0 or completed == total_blocks:
                            elapsed=time.perf_counter()-started; eta=elapsed/completed*(total_blocks-completed); print(f"progress={completed}/{total_blocks} elapsed_s={elapsed:.1f} eta_s={eta:.1f}",flush=True)
                        if completed % batches == 0:
                            _write_rows(output/"summary_metrics.partial.csv",rows); _write_rows(output/"noise_diagnostics.partial.csv",diagnostics)
    _write_rows(output/"summary_metrics.csv",rows); _write_rows(output/"noise_diagnostics.csv",diagnostics)
    unchanged=all(sha256(path)==value for path,value in hashes.items()); provenance={"torch":torch.__version__,"gpu":torch.cuda.get_device_name(0),"cuda":torch.version.cuda,"checkpoint_hashes":hashes,"checkpoint_hashes_unchanged":unchanged,"parameters":{"seeds":list(seeds),"delays":list(delays),"strengths":list(strengths),"replicates":replicates,"batches":batches,"batch_size":batch_size,"decoder_trials":decoder_trials,"phases":list(phases),"variant_chunk_size":variant_chunk_size},"rows":len(rows)}
    (output/"run_provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8"); return provenance


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/yang_fixation_circular_working_memory.yaml"); p.add_argument("--seeds",nargs="+",type=int,default=[20260714,20260715,20260716,20260717,20260718]); p.add_argument("--delays",nargs="+",type=int,default=[20,80,160]); p.add_argument("--strengths",nargs="+",type=float,default=[0,.01,.025,.05,.1]); p.add_argument("--replicates",type=int,default=5); p.add_argument("--batches",type=int,default=20); p.add_argument("--batch-size",type=int,default=64); p.add_argument("--decoder-trials",type=int,default=512); p.add_argument("--phases",nargs="+",default=["delay"]); p.add_argument("--variant-chunk-size",type=int,default=4); p.add_argument("--output",type=Path,default=Path("outputs/yang_fixation_circular_working_memory/perturbation_experiments/noise_structure")); a=p.parse_args(); result=run_experiment(load_config(a.config),a.seeds,a.delays,a.strengths,a.replicates,a.batches,a.batch_size,a.decoder_trials,a.phases,a.output,variant_chunk_size=a.variant_chunk_size); print(json.dumps(result,indent=2))


if __name__ == "__main__": main()
