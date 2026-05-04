import torch

class MyRAdam(torch.optim.Optimizer):
    def __init__(
            self, 
            params, 
            lr=1e-3, 
            betas=(0.9,0.999), 
            eps=1e-8, 
            weight_decay=0.0,
            decoupled_weight_decay=False,
    ):
        defaults = {
            "lr": lr, 
            "betas": betas, 
            "eps": eps, 
            "weight_decay": weight_decay,
            "decoupled_weight_decay": decoupled_weight_decay,
        }
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            decoupled_weight_decay = group["decoupled_weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")
                
                grad = p.grad

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                step = state["step"]

                if weight_decay != 0:
                    if decoupled_weight_decay:
                        p.mul_(1 - lr * weight_decay)
                    else:
                        grad = grad.add(p, alpha=weight_decay)
                
                # exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                beta1_t = beta1 ** step
                beta2_t = beta2 ** step
                bias_correction1 = 1 - beta1_t
                bias_correction2 = 1 - beta2_t

                bias_corrected_exp_avg = exp_avg / bias_correction1

                rho_inf = 2 / (1 - beta2) - 1
                rho_t = rho_inf - 2 * step * (beta2_t) / bias_correction2

                if rho_t > 5:
                    rect = (
                        (rho_t - 4)
                        * (rho_t - 2)
                        * rho_inf
                        / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
                    ) ** 0.5

                    exp_avg_sq_sqrt = exp_avg_sq.sqrt()
                    exp_avg_sq_sqrt = exp_avg_sq_sqrt.add_(eps)

                    adaptive_lr = (bias_correction2**0.5) / exp_avg_sq_sqrt

                    p.add_(
                        bias_corrected_exp_avg
                        * lr
                        * adaptive_lr
                        * rect,
                        alpha=-1.0,
                    )
                else:
                    p.add_(bias_corrected_exp_avg * lr, alpha=-1.0)

        return loss
