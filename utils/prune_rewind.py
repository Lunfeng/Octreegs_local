import collections
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import torch


@dataclass
class PrunedSubtreePack:
    node_ids: torch.Tensor
    topo_info: Dict[str, torch.Tensor]
    params: Dict[str, torch.Tensor]
    stats: Dict[str, torch.Tensor]
    opt_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None

    def __post_init__(self):
        self.opt_state = self.opt_state or {}
        self.size_bytes = 0
        for tensor in self.params.values():
            self.size_bytes += tensor.element_size() * tensor.nelement()
        for tensor in self.stats.values():
            self.size_bytes += tensor.element_size() * tensor.nelement()
        for tensors in self.opt_state.values():
            for tensor in tensors.values():
                self.size_bytes += tensor.element_size() * tensor.nelement()


class RewindBuffer:
    def __init__(self, budget_mb: int = 512):
        self.queue: Deque[PrunedSubtreePack] = collections.deque()
        self.budget = budget_mb * 1024 ** 2
        self.bytes = 0

    def push(self, pack: PrunedSubtreePack) -> List[PrunedSubtreePack]:
        evicted: List[PrunedSubtreePack] = []
        if pack is None:
            return evicted
        while self.bytes + pack.size_bytes > self.budget and self.queue:
            old = self.queue.popleft()
            self.bytes -= old.size_bytes
            evicted.append(old)
        if self.bytes + pack.size_bytes <= self.budget:
            self.queue.append(pack)
            self.bytes += pack.size_bytes
        else:
            evicted.append(pack)
        return evicted

    def pop_last(self) -> Optional[PrunedSubtreePack]:
        if not self.queue:
            return None
        pack = self.queue.pop()
        self.bytes -= pack.size_bytes
        return pack

    def __len__(self) -> int:
        return len(self.queue)


class PruneAndRewindManager:
    def __init__(
        self,
        model,
        buffer_budget_mb: int = 256,
        delta_psnr: float = 0.15,
        delta_lpips: float = 0.003,
        loss_window: int = 300,
        loss_delta_scale: float = 1.5,
        beta: float = 0.1,
        cooldown_steps: int = 500,
        prune_decay: float = 0.8,
    ) -> None:
        self.model = model
        self.buffer = RewindBuffer(buffer_budget_mb)
        self.delta_psnr = delta_psnr
        self.delta_lpips = delta_lpips
        self.loss_window = loss_window
        self.loss_delta_scale = loss_delta_scale
        self.beta = beta
        self.cooldown_steps = cooldown_steps
        self.prune_decay = prune_decay

        self.pending_events: List[Dict] = []
        self.loss_history: Deque[Dict[str, float]] = collections.deque(maxlen=loss_window * 2)
        self.latest_eval_metrics: Optional[Dict[str, float]] = None
        self.cooldown_registry: Dict[int, int] = {}
        self.min_opacity_scale = 1.0
        self.max_prune_nodes: Optional[int] = None

    def snapshot(self, model, prune_mask: torch.Tensor) -> Optional[PrunedSubtreePack]:
        if prune_mask is None or prune_mask.numel() == 0 or not torch.any(prune_mask):
            return None
        node_ids = model.get_anchor_uuid[prune_mask].detach().cpu()
        topo_info = {
            "level": model.get_level[prune_mask].detach().cpu(),
            "extra_level": model.get_extra_level[prune_mask].detach().cpu(),
            "uuid": node_ids.clone(),
        }
        params = {
            "anchor": model.get_anchor[prune_mask].detach().cpu(),
            "anchor_feat": model.get_anchor_feat[prune_mask].detach().cpu(),
            "offset": model._offset[prune_mask].detach().cpu(),
            "scaling": model._scaling[prune_mask].detach().cpu(),
            "rotation": model._rotation[prune_mask].detach().cpu(),
            "opacity": model._opacity[prune_mask].detach().cpu(),
        }
        stats = {
            "opacity_accum": model.opacity_accum[prune_mask].detach().cpu(),
            "anchor_demon": model.anchor_demon[prune_mask].detach().cpu(),
        }
        offset_mask = prune_mask.unsqueeze(1).repeat(1, model.n_offsets).view(-1)
        stats["offset_gradient_accum"] = model.offset_gradient_accum[offset_mask].detach().cpu()
        stats["offset_denom"] = model.offset_denom[offset_mask].detach().cpu()

        opt_state: Dict[str, Dict[str, torch.Tensor]] = {}
        for group in model.optimizer.param_groups:
            name = group.get("name", "")
            if name not in params:
                continue
            param = group["params"][0]
            state = model.optimizer.state.get(param, None)
            if state is None:
                continue
            exp_avg = state.get("exp_avg")
            exp_avg_sq = state.get("exp_avg_sq")
            if exp_avg is None or exp_avg_sq is None:
                continue
            opt_state[name] = {
                "exp_avg": exp_avg[prune_mask].detach().cpu(),
                "exp_avg_sq": exp_avg_sq[prune_mask].detach().cpu(),
            }

        return PrunedSubtreePack(node_ids=node_ids, topo_info=topo_info, params=params, stats=stats, opt_state=opt_state)

    def apply_prune_policy(self, model, prune_mask: torch.Tensor, iteration: int) -> torch.Tensor:
        if prune_mask is None or prune_mask.numel() == 0:
            return prune_mask
        mask = prune_mask.clone()
        if self.cooldown_steps > 0 and self.cooldown_registry:
            expired = [key for key, value in self.cooldown_registry.items() if value <= iteration]
            for key in expired:
                self.cooldown_registry.pop(key, None)
            anchor_uuid_cpu = model.get_anchor_uuid.detach().cpu()
            protect = torch.tensor(
                [iteration < self.cooldown_registry.get(int(uid.item()), -1) for uid in anchor_uuid_cpu],
                dtype=torch.bool,
                device=mask.device,
            )
            mask = mask & ~protect
        if self.max_prune_nodes is not None:
            idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            if idx.numel() > self.max_prune_nodes:
                mask[idx[self.max_prune_nodes :]] = False
        return mask

    def on_prune_committed(self, pack: Optional[PrunedSubtreePack], iteration: int, loss_baseline: float) -> None:
        if pack is None:
            return
        evicted = self.buffer.push(pack)
        if evicted:
            self._prune_evicted_events(evicted)
            if any(pack is ev for ev in evicted):
                return
        event = {
            "pack": pack,
            "iteration": iteration,
            "loss_baseline": loss_baseline,
            "metrics": self.latest_eval_metrics.copy() if self.latest_eval_metrics else None,
            "pruned_count": int(pack.node_ids.numel()),
        }
        self.pending_events.append(event)

    def update_iteration(self, iteration: int, loss_value: float, ema_loss: float, metrics: Optional[Dict[str, float]] = None) -> bool:
        self.loss_history.append({"iteration": iteration, "loss": loss_value, "ema": ema_loss})
        if metrics:
            self.latest_eval_metrics = metrics
        return self._maybe_rewind(iteration)

    def _maybe_rewind(self, iteration: int) -> bool:
        if not self.pending_events:
            return False
        current_metrics = self.latest_eval_metrics or {}
        event = self.pending_events[-1]
        if self._triggered_by_metrics(event, current_metrics):
            return self._rewind(iteration, event)
        if self._triggered_by_loss(event):
            return self._rewind(iteration, event)
        return False

    def _triggered_by_metrics(self, event: Dict, metrics: Dict[str, float]) -> bool:
        baseline = event.get("metrics") or {}
        if not baseline or not metrics:
            return False
        psnr_cur = metrics.get("psnr")
        psnr_prev = baseline.get("psnr")
        if psnr_cur is not None and psnr_prev is not None:
            if psnr_cur - psnr_prev <= -self.delta_psnr:
                return True
        lpips_cur = metrics.get("lpips")
        lpips_prev = baseline.get("lpips")
        if lpips_cur is not None and lpips_prev is not None:
            if lpips_cur - lpips_prev >= self.delta_lpips:
                return True
        return False

    def _triggered_by_loss(self, event: Dict) -> bool:
        if not self.loss_history:
            return False
        losses_after = [item["ema"] for item in self.loss_history if item["iteration"] > event["iteration"]]
        if len(losses_after) < self.loss_window:
            return False
        recent = torch.tensor(losses_after[-self.loss_window :], dtype=torch.float32)
        q1 = torch.quantile(recent, 0.25)
        q3 = torch.quantile(recent, 0.75)
        iqr = (q3 - q1).item()
        delta_loss = self.loss_delta_scale * max(iqr, 1e-6)
        baseline = event.get("loss_baseline", 0.0)
        if torch.all(recent >= baseline + delta_loss):
            return True
        return False

    def _rewind(self, iteration: int, event: Dict) -> bool:
        pack = self.buffer.pop_last()
        if pack is None:
            self.pending_events.pop()
            return False
        self.model.restore_pruned_pack(pack)
        print(f"[PruneAndRewind] Rewinding {pack.node_ids.numel()} anchors at iter {iteration}")
        self.pending_events.pop()
        self._register_cooldown(pack.node_ids, iteration)
        self.min_opacity_scale *= 1.0 + self.beta
        if event.get("pruned_count"):
            new_limit = max(1, int(event["pruned_count"] * self.prune_decay))
            if self.max_prune_nodes is None:
                self.max_prune_nodes = new_limit
            else:
                self.max_prune_nodes = min(self.max_prune_nodes, new_limit)
        return True

    def _register_cooldown(self, node_ids: torch.Tensor, iteration: int) -> None:
        if self.cooldown_steps <= 0:
            return
        for node_id in node_ids:
            self.cooldown_registry[int(node_id.item())] = iteration + self.cooldown_steps
        expired = [key for key, value in self.cooldown_registry.items() if value <= iteration]
        for key in expired:
            self.cooldown_registry.pop(key, None)

    def _prune_evicted_events(self, evicted_packs: List[PrunedSubtreePack]) -> None:
        if not evicted_packs:
            return
        remaining_events = []
        evicted_set = set(id(pack) for pack in evicted_packs)
        for event in self.pending_events:
            if id(event.get("pack")) in evicted_set:
                continue
            remaining_events.append(event)
        self.pending_events = remaining_events 
