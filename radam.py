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
            params = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            steps = []

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

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                params.append(p)
                grads.append(p.grad)
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                steps.append(state["step"])

            # --- update ---
            _single_radam_step(
                params,
                grads,
                exp_avgs,
                exp_avg_sqs,
                steps,
                weight_decay,
                decoupled_weight_decay,
                lr,
                beta1,
                beta2,
                eps,
            )

        return loss

def _single_radam_step(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    steps,
    weight_decay,
    decoupled_weight_decay,
    lr,
    beta1,
    beta2,
    eps,
):
    for i, p in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]

        # --- step ---
        steps[i].add_(1)
        step = steps[i] # tensor
        # step = steps[i].item() # scaler
        # step = float(steps[i]) # mps

        # --- weight decay ---
        if weight_decay != 0:
            if decoupled_weight_decay:
                p.mul_(1 - lr * weight_decay)
            else:
                grad = grad.add(p, alpha=weight_decay)

        # 移動平均
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # --- bias correction ---
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        # 自由度を計算
        rho_inf = 2 / (1 - beta2) - 1
        rho_t = rho_inf - 2 * step * (beta2**step) / bias_correction2

        if rho_t > 4:
            # 補正項付きAdam
            # p -= lr * rect * m_hat / {sqrt(v_hat) + eps}
            rect = (
                (rho_t - 4)
                * (rho_t - 2)
                * rho_inf
                / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
            ) ** 0.5

            bias_correction2_sqrt = bias_correction2 ** 0.5

            # denom = exp_avg_sq.sqrt().div_(bias_correction2_sqrt)
            # denom.add_(eps)
            denom = exp_avg_sq.sqrt().div(bias_correction2_sqrt).add_(eps)

            step_size = lr * rect / bias_correction1

            p.addcdiv_(exp_avg, denom, value=-step_size)

        else:
            # モーメントつきSGD
            # p -= lr * m_hat
            step_size = lr / bias_correction1
            p.add_(exp_avg, alpha=-step_size)
