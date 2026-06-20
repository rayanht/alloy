"""LoRA fine-tuning of gpt2 through the Alloy backend. Needs the transformers + peft extras."""

import torch
import torch.nn.functional as F
import transformers
import peft
import alloy_torch  # noqa: F401  imports register the "alloy" backend
from alloy_torch.training import set_training_mode

set_training_mode(True)


def main() -> None:
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    text = "Alloy compiles PyTorch kernels to Metal for Apple Silicon."
    ids = tok(text, return_tensors="pt").input_ids

    torch.manual_seed(0)
    base = transformers.AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager")
    cfg = peft.LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["c_attn", "c_proj", "c_fc"], task_type="CAUSAL_LM",
    )
    model = peft.get_peft_model(base, cfg)
    step = torch.compile(model, backend="alloy")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-3)

    model.train()
    first = last = 0.0
    for i in range(100):
        opt.zero_grad()
        logits = step(input_ids=ids).logits
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        last = float(loss.detach())
        if i == 0:
            first = last
    print(f"LoRA fine-tune (100 steps): loss {first:.3f} -> {last:.3f}")

    model.eval()
    cur = ids[:, :4].clone()
    for _ in range(ids.shape[1] - 4):
        cur = torch.cat([cur, step(input_ids=cur).logits[:, -1].argmax(-1, keepdim=True)], 1)
    print("prompt:   ", repr(tok.decode(ids[0, :4])))
    print("generated:", repr(tok.decode(cur[0])))

    assert tok.decode(cur[0]) == text, "fine-tuned model did not reproduce the text"
    print("PASSED")


if __name__ == "__main__":
    main()
