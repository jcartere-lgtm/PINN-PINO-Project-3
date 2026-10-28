"""A small physics-informed DeepONet pilot for the one-dimensional GPE.

This is a branch/trunk DeepONet, not an implementation of the published Fourier
Neural Operator (FNO/PINO). The branch sees an entire piecewise-linear control;
the trunk sees position and time. The two outputs are Re(psi) and Im(psi).

The residual is for i psi_t = -psi_xx/2 + (u(t) x**2/2 + g |psi|**2) psi.
Both data and physics losses use *complex* states. No prediction is normalized
afterward. All losses in the returned history are unweighted raw means; the
training objective uses the documented weights and residual_scale below.

Only NumPy and PyTorch are needed. Importing this module does not train a model
or write any file.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn


def _mlp(n_input, n_output, width, depth):
    layers = []
    previous = n_input
    for _ in range(depth):
        layers.extend((nn.Linear(previous, width), nn.Tanh()))
        previous = width
    layers.append(nn.Linear(previous, n_output))
    network = nn.Sequential(*layers)
    for layer in network:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    return network


class ComplexDeepONet(nn.Module):
    """Control-function to complex-field map with explicit periodic BC loss.

    Coordinates are supplied as physical ``[x,t]`` pairs. Raw x is included in
    addition to Fourier features, so periodicity is *not* hard imposed. A value
    and first-derivative boundary loss supplies periodic boundary conditions.
    Control normalization is fixed (default [1,100]), never fitted to test data.
    """

    def __init__(self, knot_times, config=None):
        super().__init__()
        cfg = dict(config or {})
        self.length = float(cfg.get("length", 16.0))
        self.T = float(cfg.get("T", 2.5))
        self.g = float(cfg.get("g", 1.0))
        self.u_min = float(cfg.get("u_min", 1.0))
        self.u_max = float(cfg.get("u_max", 100.0))
        if self.length <= 0 or self.T <= 0 or self.u_max <= self.u_min:
            raise ValueError("length, T, and control normalization range must be positive")
        self.latent = int(cfg.get("latent", 32))
        width = int(cfg.get("width", 64))
        depth = int(cfg.get("depth", 2))
        shared_frequencies = cfg.get("frequencies", (1, 2, 4, 8))
        sf = cfg.get("space_frequencies", shared_frequencies)
        tf = cfg.get("time_frequencies", shared_frequencies)
        self.register_buffer("space_frequencies", torch.as_tensor(sf, dtype=torch.float32))
        self.register_buffer("time_frequencies", torch.as_tensor(tf, dtype=torch.float32))
        knot_times = torch.as_tensor(knot_times, dtype=torch.float32)
        if knot_times.ndim != 1 or knot_times.numel() < 2:
            raise ValueError("At least two control knot times are required")
        if not bool(torch.all(knot_times[1:] > knot_times[:-1])):
            raise ValueError("Control knot times must be strictly increasing")
        if abs(float(knot_times[0])) > 1e-6 or abs(float(knot_times[-1]) - self.T) > 1e-6:
            raise ValueError("Control knots must span [0,T]")
        self.register_buffer("knot_times", knot_times)
        n_features = 2 + 2 * len(sf) + 2 * len(tf)
        self.branch = _mlp(knot_times.numel(), 2 * self.latent, width, depth)
        self.trunk = _mlp(n_features, self.latent, width, depth)
        self.bias = nn.Parameter(torch.zeros(2))
        self.config = cfg

    def forward(self, knots, coordinates):
        if knots.ndim == 1:
            knots = knots.unsqueeze(0).expand(coordinates.shape[0], -1)
        branch_input = 2 * (knots - self.u_min) / (self.u_max - self.u_min) - 1
        x = coordinates[:, :1]
        t = coordinates[:, 1:2]
        sx = 2 * torch.pi * x / self.length * self.space_frequencies
        st = 2 * torch.pi * t / self.T * self.time_frequencies
        features = torch.cat(
            (2 * x / self.length, 2 * t / self.T - 1,
             torch.sin(sx), torch.cos(sx), torch.sin(st), torch.cos(st)), dim=1
        )
        branch = self.branch(branch_input).reshape(-1, 2, self.latent)
        trunk = self.trunk(features).unsqueeze(1)
        return (branch * trunk).sum(dim=-1) + self.bias

    def control_at(self, knots, t):
        """Differentiable-in-value interpolation of the complete knot control."""
        if knots.ndim == 1:
            knots = knots.unsqueeze(0).expand(t.numel(), -1)
        t = t.reshape(-1).clamp(0.0, self.T)
        segment = torch.searchsorted(self.knot_times, t.contiguous(), right=True) - 1
        segment = segment.clamp(0, self.knot_times.numel() - 2)
        left = self.knot_times[segment]
        right = self.knot_times[segment + 1]
        fraction = (t - left) / (right - left)
        first = knots.gather(1, segment[:, None]).squeeze(1)
        second = knots.gather(1, (segment + 1)[:, None]).squeeze(1)
        return first + fraction * (second - first)


def physics_residual(model, knots, coordinates):
    """Return real/imaginary components of i psi_t + psi_xx/2 - W psi.

    The derivative graph is retained so the residual can train the network.
    ``coordinates`` is a tensor of physical [x,t] rows; a fresh differentiable
    copy is used if it does not already require gradients.
    """
    if not coordinates.requires_grad:
        coordinates = coordinates.detach().clone().requires_grad_(True)
    state = model(knots, coordinates)
    a, b = state[:, 0], state[:, 1]
    grad_a = torch.autograd.grad(a.sum(), coordinates, create_graph=True)[0]
    grad_b = torch.autograd.grad(b.sum(), coordinates, create_graph=True)[0]
    a_xx = torch.autograd.grad(grad_a[:, 0].sum(), coordinates, create_graph=True)[0][:, 0]
    b_xx = torch.autograd.grad(grad_b[:, 0].sum(), coordinates, create_graph=True)[0][:, 0]
    potential = 0.5 * model.control_at(knots, coordinates[:, 1]) * coordinates[:, 0] ** 2
    effective = potential + model.g * (a.square() + b.square())
    real = -grad_b[:, 1] + 0.5 * a_xx - effective * a
    imag = grad_a[:, 1] + 0.5 * b_xx - effective * b
    return torch.stack((real, imag), dim=1)


def _complex_mse(predicted, target):
    return (predicted - target).square().sum(dim=-1).mean()


def _boundary_loss(model, knots, times):
    n_points = times.numel()
    left = torch.stack((-torch.ones_like(times) * model.length / 2, times), dim=1)
    right = torch.stack((torch.ones_like(times) * model.length / 2, times), dim=1)
    both = torch.cat((left, right), dim=0).requires_grad_(True)
    states = model(torch.cat((knots, knots), dim=0), both)
    value_loss = (states[:n_points] - states[n_points:]).square().sum(dim=1).mean()
    derivatives = []
    for part in range(2):
        gradient = torch.autograd.grad(states[:, part].sum(), both, create_graph=True)[0]
        derivatives.append(gradient[:, 0])
    derivative = torch.stack(derivatives, dim=1)
    derivative_loss = (
        derivative[:n_points] - derivative[n_points:]
    ).square().sum(dim=1).mean()
    return value_loss + derivative_loss, value_loss, derivative_loss


def predict_operator(model, knots, x, t, batch_size=65536):
    """Predict one full complex trajectory, shape (len(t),len(x)).

    This function returns the raw network output; mass is not corrected.
    """
    x = np.asarray(x, dtype=np.float32)
    t = np.asarray(t, dtype=np.float32)
    xx, tt = np.meshgrid(x, t, indexing="xy")
    coordinates = np.column_stack((xx.ravel(), tt.ravel()))
    device = next(model.parameters()).device
    knot_tensor = torch.as_tensor(knots, dtype=torch.float32, device=device)
    predictions = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, len(coordinates), batch_size):
            coords = torch.as_tensor(coordinates[start:start + batch_size], device=device)
            predictions.append(model(knot_tensor, coords).cpu().numpy())
    model.train(was_training)
    values = np.concatenate(predictions, axis=0)
    return (values[:, 0] + 1j * values[:, 1]).reshape(len(t), len(x))


def evaluate_operator(model, dataset, indices):
    """Whole-control metrics, with every time/space point in the supplied data."""
    indices = np.asarray(indices, dtype=int)
    if not indices.size:
        return {"relative_l2": None, "density_relative_l2": None,
                "mass_max_abs_error": None, "n_controls": 0}
    x = np.asarray(dataset["x"])
    dx = float(x[1] - x[0])
    error = truth_norm = density_error = density_norm = 0.0
    max_mass_error = 0.0
    for control_index in indices:
        predicted = predict_operator(
            model, dataset["knot_values"][control_index], x, dataset["t"]
        )
        truth = np.asarray(dataset["psi"])[control_index]
        error += float(np.sum(np.abs(predicted - truth) ** 2))
        truth_norm += float(np.sum(np.abs(truth) ** 2))
        density_error += float(np.sum((np.abs(predicted) ** 2 - np.abs(truth) ** 2) ** 2))
        density_norm += float(np.sum(np.abs(truth) ** 4))
        masses = np.sum(np.abs(predicted) ** 2, axis=1) * dx
        max_mass_error = max(max_mass_error, float(np.max(np.abs(masses - 1.0))))
    return {
        "relative_l2": float(np.sqrt(error / max(truth_norm, 1e-30))),
        "density_relative_l2": float(np.sqrt(density_error / max(density_norm, 1e-30))),
        "mass_max_abs_error": max_mass_error,
        "n_controls": int(indices.size),
    }


def fit_operator(dataset, train_indices, val_indices, config=None,
                 mode="physics", steps=1000, device="cpu"):
    """Fit the pilot using a split by *entire controls*, never by grid points.

    Required dataset keys: x, t, psi[control,time,space] (complex), knot_values.
    Optional: knot_times and psi0[space]. All controls must share psi0. A final
    test split must be withheld by the caller and must not appear in either
    supplied index list. Validation metrics monitor fitting only; no early
    stopping or checkpoint selection is performed.

    Defaults: width=64, depth=2, latent=32, Adam lr=1e-3, data/PDE batches=256,
    initial/boundary batches=128/32. Raw complex MSE sums Re/Im errors. The
    objective is wd*data + wi*IC + wp*PDE/residual_scale**2 + wb*BC + wn*norm,
    with default weights data=1, ic=1, pde=1, bc=.1, norm=0 and
    residual_scale=100. This constant scale prevents the unbounded harmonic
    potential at distant grid points from swamping data during the pilot; the
    unscaled PDE MSE is still logged. These are training choices, not physical
    coefficients. Boundary loss sums periodic value and x-derivative errors.
    By default half of data/IC samples use grid points in |x|<=3 and half use
    the full grid. PDE points use the same mixture of continuous uniforms.
    Configure central_fraction, central_radius, or pde_central_fraction to
    change this deliberate spatial weighting; validation uses the full grid.

    mode='supervised' uses the identical architecture, seed, random data/IC
    samples, steps and optimizer, while removing PDE/BC/norm terms. IC samples
    are the same supplied initial-state labels in both modes. Skipped losses
    are logged as None, never as an artificial zero. Logs contain every step;
    validation is measured at log_every (default 100) and at the final step.
    """
    if mode not in ("physics", "supervised"):
        raise ValueError("mode must be 'physics' or 'supervised'")
    cfg = dict(config or {})
    torch.set_num_threads(int(cfg.get("num_threads", 2)))
    seed = int(cfg.get("seed", 598))
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    train_indices = np.asarray(train_indices, dtype=int)
    val_indices = np.asarray(val_indices, dtype=int)
    if train_indices.size == 0 or np.intersect1d(train_indices, val_indices).size:
        raise ValueError("Training controls must be nonempty and disjoint from validation controls")
    x = np.asarray(dataset["x"], dtype=np.float32)
    t = np.asarray(dataset["t"], dtype=np.float32)
    psi = np.asarray(dataset["psi"])
    knots = np.asarray(dataset["knot_values"], dtype=np.float32)
    if psi.shape != (knots.shape[0], t.size, x.size):
        raise ValueError("psi must have shape (n_controls,len(t),len(x))")
    if x.size < 2 or t.size < 2 or not np.iscomplexobj(psi):
        raise ValueError("At least two x/t samples and complex-valued psi are required")
    all_indices = np.concatenate((train_indices, val_indices))
    if np.any(all_indices < 0) or np.any(all_indices >= len(knots)):
        raise ValueError("Control index outside dataset")
    cfg.setdefault("T", float(t[-1]))
    cfg.setdefault("length", float((x[1] - x[0]) * x.size))
    knot_times = np.asarray(dataset.get("knot_times", np.linspace(0, cfg["T"], knots.shape[1])))
    psi0 = np.asarray(dataset.get("psi0", psi[train_indices[0], 0]))
    if psi0.shape != (x.size,):
        raise ValueError("psi0 must have shape (len(x),)")
    if not np.allclose(psi[train_indices, 0], psi0, atol=2e-6, rtol=2e-5):
        raise ValueError("This pilot assumes one shared initial state across training controls")
    model = ComplexDeepONet(knot_times, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 1e-3)))
    loss_weights = {"data": 1.0, "ic": 1.0, "pde": 1.0, "bc": 0.1, "norm": 0.0}
    loss_weights.update(cfg.get("loss_weights", {}))
    residual_scale = float(cfg.get("residual_scale", 100.0))
    if residual_scale <= 0:
        raise ValueError("residual_scale must be positive")
    batches = {"data": int(cfg.get("batch_data", 256)),
               "pde": int(cfg.get("batch_pde", 256)),
               "ic": int(cfg.get("batch_ic", 128)),
               "bc": int(cfg.get("batch_bc", 32)),
               "norm": int(cfg.get("batch_norm", 2))}
    if any(value <= 0 for value in batches.values()) or int(steps) < 1:
        raise ValueError("Batch sizes and steps must be positive")
    log_every = max(1, int(cfg.get("log_every", 100)))
    central_fraction = float(cfg.get("central_fraction", 0.5))
    pde_central_fraction = float(cfg.get("pde_central_fraction", central_fraction))
    central_radius = min(float(cfg.get("central_radius", 3.0)), model.length / 2)
    if not (0 <= central_fraction <= 1 and 0 <= pde_central_fraction <= 1 and central_radius > 0):
        raise ValueError("Spatial mixture fractions must be in [0,1] and radius positive")
    central_indices = np.flatnonzero(np.abs(x) <= central_radius)
    if central_indices.size == 0:
        raise ValueError("The central sampling interval contains no grid points")
    knots_tensor = torch.as_tensor(knots, device=device)

    def tensor(array):
        return torch.as_tensor(array, dtype=torch.float32, device=device)

    def random_controls(n_points):
        return train_indices[rng.integers(0, train_indices.size, n_points)]

    def sample_grid_indices(n_points):
        n_central = int(round(n_points * central_fraction))
        selected = np.concatenate((rng.choice(central_indices, n_central),
                                   rng.integers(0, x.size, n_points - n_central)))
        return rng.permutation(selected)

    def sample_physics_positions(n_points):
        n_central = int(round(n_points * pde_central_fraction))
        selected = np.concatenate((rng.uniform(-central_radius, central_radius, n_central),
                                   rng.uniform(-model.length / 2, model.length / 2,
                                               n_points - n_central)))
        return rng.permutation(selected)

    # A fixed validation sample uses a separate generator, preserving identical
    # training samples when validation size or log frequency changes.
    val_rng = np.random.default_rng(seed + 1)
    if val_indices.size:
        nv = min(int(cfg.get("validation_points", 2048)), val_indices.size * t.size * x.size)
        vc = val_rng.choice(val_indices, nv)
        vt = val_rng.integers(0, t.size, nv)
        vx = val_rng.integers(0, x.size, nv)
        validation_coords = tensor(np.column_stack((x[vx], t[vt])))
        validation_truth = tensor(np.column_stack((psi[vc, vt, vx].real, psi[vc, vt, vx].imag)))
        validation_knots = knots_tensor[torch.as_tensor(vc, device=device)]

    history = []
    start_time = time.perf_counter()
    for step in range(1, int(steps) + 1):
        # Draw every family in both modes, keeping the supervised samples equal.
        dc = random_controls(batches["data"])
        dti = rng.integers(0, t.size, batches["data"])
        dxi = sample_grid_indices(batches["data"])
        pc = random_controls(batches["pde"])
        px = sample_physics_positions(batches["pde"])
        pt = rng.uniform(0, model.T, batches["pde"])
        ic = random_controls(batches["ic"])
        ixi = sample_grid_indices(batches["ic"])
        bc = random_controls(batches["bc"])
        bt = rng.uniform(0, model.T, batches["bc"])
        nc = random_controls(batches["norm"])
        nt = rng.uniform(0, model.T, batches["norm"])
        optimizer.zero_grad(set_to_none=True)
        data_prediction = model(knots_tensor[torch.as_tensor(dc, device=device)],
                                tensor(np.column_stack((x[dxi], t[dti]))))
        observed = psi[dc, dti, dxi]
        loss_data = _complex_mse(data_prediction, tensor(np.column_stack((observed.real, observed.imag))))
        initial_prediction = model(knots_tensor[torch.as_tensor(ic, device=device)],
                                   tensor(np.column_stack((x[ixi], np.zeros_like(ixi)))))
        initial_truth = psi0[ixi]
        loss_ic = _complex_mse(initial_prediction, tensor(np.column_stack((initial_truth.real, initial_truth.imag))))
        loss_pde = loss_bc = loss_bc_value = loss_bc_derivative = loss_norm = None
        total = loss_weights["data"] * loss_data + loss_weights["ic"] * loss_ic
        if mode == "physics":
            residual = physics_residual(model, knots_tensor[torch.as_tensor(pc, device=device)],
                                        tensor(np.column_stack((px, pt))))
            loss_pde = residual.square().sum(dim=1).mean()
            loss_bc, loss_bc_value, loss_bc_derivative = _boundary_loss(
                model, knots_tensor[torch.as_tensor(bc, device=device)], tensor(bt))
            total = (total + loss_weights["pde"] * loss_pde / residual_scale**2
                     + loss_weights["bc"] * loss_bc)
            if loss_weights["norm"]:
                norm_x = np.tile(x, len(nc))
                norm_t = np.repeat(nt, x.size)
                norm_knots = knots_tensor[torch.as_tensor(np.repeat(nc, x.size), device=device)]
                norm_state = model(norm_knots, tensor(np.column_stack((norm_x, norm_t))))
                mass = norm_state.square().sum(dim=1).reshape(len(nc), x.size).sum(dim=1) * float(x[1] - x[0])
                loss_norm = (mass - 1).square().mean()
                total = total + loss_weights["norm"] * loss_norm
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(f"Nonfinite {mode} training objective at step {step}")
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("gradient_clip", 10.0)))
        optimizer.step()
        row = {"step": step, "loss_total": float(total.detach()),
               "loss_data": float(loss_data.detach()), "loss_ic": float(loss_ic.detach()),
               "loss_pde": None if loss_pde is None else float(loss_pde.detach()),
               "loss_bc": None if loss_bc is None else float(loss_bc.detach()),
               "loss_bc_value": None if loss_bc_value is None else float(loss_bc_value.detach()),
               "loss_bc_derivative": None if loss_bc_derivative is None else float(loss_bc_derivative.detach()),
               "loss_norm": None if loss_norm is None else float(loss_norm.detach()),
               "gradient_norm_before_clip": float(gradient_norm),
               "val_relative_l2_sample": None,
               "elapsed_seconds": float(time.perf_counter() - start_time)}
        if val_indices.size and (step == 1 or step % log_every == 0 or step == int(steps)):
            with torch.no_grad():
                val_prediction = model(validation_knots, validation_coords)
                relative = ((val_prediction - validation_truth).square().sum()
                            / validation_truth.square().sum().clamp_min(1e-30)).sqrt()
                row["val_relative_l2_sample"] = float(relative)
        history.append(row)
    training_seconds = float(time.perf_counter() - start_time)
    validation = evaluate_operator(model, dataset, val_indices)
    summary = {
        "architecture": "complex branch/trunk DeepONet with Fourier coordinate features",
        "mode": mode, "seed": seed, "steps": int(steps), "device": str(device),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "train_indices": train_indices.tolist(), "val_indices": val_indices.tolist(),
        "training_seconds": training_seconds,
        "loss_weights": loss_weights, "residual_scale": residual_scale,
        "spatial_sampling": {"central_fraction": central_fraction,
                             "pde_central_fraction": pde_central_fraction,
                             "central_radius": central_radius},
        "validation": validation, "prediction_renormalized": False,
        "notes": ["Pilot DeepONet; not the published FNO/PINO architecture.",
                  "Physics residual uses raw complex states, g as configured, and interpolated complete controls.",
                  "Supervised mode keeps the same data and initial-state labels while omitting physics penalties.",
                  "Raw losses are logged separately; the PDE term in total loss is divided by residual_scale squared.",
                  "Validation controls are not used for training or checkpoint selection; reserve a separate test split."]}
    return model, history, summary
