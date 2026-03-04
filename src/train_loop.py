import os
import torch
from accelerate import Accelerator


def _move_batch_to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def save_best_checkpoint(model, optimizer, accelerator: Accelerator, output_dir: str, epoch: int, best_val_loss: float, global_update: int):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    ckpt_dir = os.path.join(output_dir, "best")
    os.makedirs(ckpt_dir, exist_ok=True)

    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(ckpt_dir)

    state = {
        "optimizer": optimizer.state_dict(),
        "meta": {
            "epoch": epoch,
            "global_update": global_update,
            "best_val_loss": best_val_loss,
        },
    }
    torch.save(state, os.path.join(ckpt_dir, "training_state.pt"))


def try_resume_best(model, optimizer, accelerator: Accelerator, output_dir: str, resume_dir: str = ""):
    # pick resume path
    if resume_dir:
        ckpt_dir = resume_dir
    else:
        ckpt_dir = os.path.join(output_dir, "best")

    state_path = os.path.join(ckpt_dir, "training_state.pt")
    if not os.path.exists(state_path):
        if accelerator.is_main_process:
            print(f"[resume] No checkpoint found, start from Epoch 0")
        return 0, 0, float("inf")

    # Load adapters
    try:
        model.load_adapter(ckpt_dir, adapter_name="default", is_trainable=True)
        model.set_adapter("default")
    except Exception as e:
        if accelerator.is_main_process:
            print(f"[resume] WARNING: could not load adapter from {ckpt_dir}: {e}")
        pass

    state = torch.load(state_path, map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    meta = state.get("meta", {})

    start_epoch = int(meta.get("epoch", 0)) + 1  # resume from next epoch
    global_update = int(meta.get("global_update", 0))
    best_val = float(meta.get("best_val_loss", float("inf")))

    if accelerator.is_main_process:
        print(f"[resume] loaded {ckpt_dir} -> start_epoch={start_epoch}, global_update={global_update}, best_val={best_val}")

    return start_epoch, global_update, best_val


def train(
    model,
    optimizer,
    train_loader,
    val_loader,
    accelerator: Accelerator,
    epochs: int,
    grad_accum: int,
    output_dir: str,
    resume_dir: str,
    log_every_updates: int = 10,
):
    # optional resume
    start_epoch, global_update, best_val = try_resume_best(
        model=model,
        optimizer=optimizer,
        accelerator=accelerator,
        output_dir=output_dir,
        resume_dir=resume_dir
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, epochs):
        # -------- TRAIN --------
        model.train()
        for step, batch in enumerate(train_loader):
            batch = _move_batch_to_device(batch, accelerator.device)

            out = model(**batch)
            loss = out.loss / grad_accum
            accelerator.backward(loss)

            if (step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1

                if accelerator.is_main_process and (global_update % log_every_updates == 0):
                    print(f"[train] epoch={epoch} update={global_update} loss={(loss.item()*grad_accum):.4f}")

        # -------- VALIDATION --------
        accelerator.wait_for_everyone()
        val_loss = evaluate(model, val_loader, accelerator)

        if accelerator.is_main_process:
            print(f"[val] epoch={epoch} avg_loss={val_loss:.4f} (best={best_val:.4f})")

        # save ONLY if improved
        if val_loss < best_val:
            best_val = val_loss
            save_best_checkpoint(
                model=model,
                optimizer=optimizer,
                accelerator=accelerator,
                output_dir=output_dir,
                epoch=epoch,
                best_val_loss=best_val,
                global_update=global_update,
            )
            if accelerator.is_main_process:
                print(f"[ckpt] saved new BEST at epoch={epoch} val_loss={best_val:.4f}")

    accelerator.wait_for_everyone()
    
@torch.no_grad()
def evaluate(model, val_loader, accelerator: Accelerator):
    model.eval()
    losses = []

    for batch in val_loader:
        batch = _move_batch_to_device(batch, accelerator.device)
        out = model(**batch)
        loss = out.loss.detach()

        # gather across processes (multi-GPU)
        loss = accelerator.gather(loss)
        losses.append(loss)

    if not losses:
        return float("nan")

    all_losses = torch.cat([x.flatten() for x in losses], dim=0)
    return all_losses.mean().item()