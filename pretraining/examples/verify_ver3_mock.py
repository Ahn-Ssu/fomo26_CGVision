"""ver3 Step 1: mock-tensor verification (no real data, CPU-friendly, minutes).

Checks (per ver3 spec):
  1. Shapes flow correctly end-to-end with --convpass_peft on.
  2. up=0 / gate=0 init means ver3's output is BIT-IDENTICAL (torch.equal,
     not allclose) to ver2's, for the same shared weights. Verified by
     building ONE ver3 model, copying its shared (non-Convpass) weights into
     a freshly-built ver2 model, then comparing forward_with_features()
     output tensors exactly. (Building two SEPARATELY-seeded models would
     NOT give identical shared weights past decoder stage 0 -- ver3's extra
     per-stage Convpass kaiming_normal_ init calls consume RNG state that a
     ver2 model never does, so stage 1+ would silently diverge. Copying
     state_dict sidesteps that entirely.)
  3. Teacher switching actually activates different parameters -- selecting
     a different teacher_name changes the output, and only that teacher's
     gate/alpha/convpass params receive gradient on backward.
"""

import sys

import torch

sys.path.insert(0, "/root")
from FOMO26.networks.student import build_student  # noqa: E402

TEACHERS = ["anatomix+brains", "vesselfm", "brats"]


def check_bit_identical():
    print("=== Check 2: up=0/gate=0 init -> bit-identical to ver2 ===")
    torch.manual_seed(0)
    ver3 = build_student(in_channels=1, teachers=TEACHERS, convpass=True)
    ver3.eval()

    ver2 = build_student(in_channels=1, teachers=TEACHERS, convpass=False)
    ver3_sd = ver3.state_dict()
    shared_sd = {k: v for k, v in ver3_sd.items()
                 if ".skip_alpha." not in k and ".convpass_gate." not in k and ".convpass." not in k}
    missing, unexpected = ver2.load_state_dict(shared_sd, strict=False)
    assert not unexpected, f"unexpected keys not consumed by ver2: {unexpected}"
    assert not missing, f"ver2 has keys with no match in ver3's shared set: {missing}"
    ver2.eval()

    x = torch.randn(1, 1, 64, 64, 64)
    for teacher in TEACHERS:
        with torch.no_grad():
            feats3 = ver3.forward_with_features(x, teacher)
            feats2 = ver2.forward_with_features(x, teacher)
        for k in feats2:
            identical = torch.equal(feats2[k], feats3[k])
            print(f"  teacher={teacher} {k}: bit-identical={identical}")
            assert identical, f"MISMATCH at teacher={teacher} {k} -- Convpass insertion is NOT a no-op at init!"
    print("  PASSED: ver3 (fresh init) == ver2 (same shared weights) exactly, all teachers/stages.\n")
    return ver3


def check_teacher_switch_changes_output(ver3):
    print("=== Check 3a: different teacher_name -> different output (once gates are non-zero) ===")
    # Force non-zero gates/alphas so a real difference is possible to observe
    # (at init, gate=0 for ALL teachers means outputs would be identical
    # across teachers too -- that's expected/correct at init, not a bug, so
    # we perturb gates here specifically to test the SWITCHING mechanism).
    with torch.no_grad():
        for stage in ver3.decoder.stages:
            if stage.convpass_enabled:
                for t in TEACHERS:
                    stage.convpass_gate[t].fill_(0.5)
                    stage.skip_alpha[t].fill_(1.0 + 0.1 * TEACHERS.index(t))
                    for p in stage.convpass[t].parameters():
                        if p.dim() > 1:
                            torch.nn.init.kaiming_normal_(p, nonlinearity="relu")

    x = torch.randn(1, 1, 64, 64, 64)
    outs = {}
    with torch.no_grad():
        for teacher in TEACHERS:
            outs[teacher] = ver3.forward_with_features(x, teacher)["dec_stage_4"].clone()
    for i in range(len(TEACHERS)):
        for j in range(i + 1, len(TEACHERS)):
            t1, t2 = TEACHERS[i], TEACHERS[j]
            different = not torch.equal(outs[t1], outs[t2])
            print(f"  {t1} vs {t2}: different={different}")
            assert different, f"teacher switching had NO effect: {t1} and {t2} gave identical output"
    print("  PASSED: switching teacher_name changes the forward output once gates are non-zero.\n")


def check_gradient_isolation(ver3):
    print("=== Check 3b: only the SAMPLED teacher's gate/alpha/convpass params get gradient ===")
    ver3.train()
    x = torch.randn(2, 1, 64, 64, 64, requires_grad=False)
    sampled_teacher = "brats"
    feats = ver3.forward_with_features(x, sampled_teacher)
    loss = sum(f.pow(2).mean() for f in feats.values())
    loss.backward()

    gate_params = {n: p for n, p in ver3.named_parameters() if ".convpass_gate." in n or ".skip_alpha." in n}
    for name, p in gate_params.items():
        teacher_in_name = next(t for t in TEACHERS if name.endswith(t))
        has_grad = p.grad is not None and p.grad.abs().sum().item() > 0
        expected = (teacher_in_name == sampled_teacher)
        status = "OK" if has_grad == expected else "MISMATCH"
        if status == "MISMATCH":
            print(f"  {status}: {name} has_grad={has_grad} expected={expected}")
    mismatches = [n for n, p in gate_params.items()
                  if (p.grad is not None and p.grad.abs().sum().item() > 0) != n.endswith(sampled_teacher)]
    assert not mismatches, f"gradient isolation broken for: {mismatches}"
    print(f"  PASSED: only '{sampled_teacher}' gate/alpha params received gradient "
          f"({sum(1 for n in gate_params if n.endswith(sampled_teacher))} params); "
          f"other {len(TEACHERS)-1} teachers' gate/alpha params stayed at grad=None/0.\n")


def check_convpass_gradient_not_dead():
    """Regression guard for the 2026-07-24 double-zero-init bug: the
    ORIGINAL ver3 spec had both `up` and `convpass_gate` zero-initialized,
    which is a genuine dead fixed point -- d(out)/d(gate) = up(...)'s
    current output, which is identically 0 whenever up.weight=0, so gate
    NEVER receives gradient, and down/dw/up (gated multiplicatively by
    gate=0) never do either -- a permanent fixed point, not slow learning.

    Fix (applied 2026-07-24): Kaiming-init `up` instead of zeros_ (gate=0
    alone already guarantees the no-op-at-init property, so zeroing `up`
    too was redundant and is what created the trap). This produces a
    staggered two-step wake-up, mirroring standard LoRA's B-then-A pattern:
      step 1: gate gets NONZERO gradient (d(out)/d(gate) = up(dw(down(x)))
              is now a real nonzero tensor); down/dw/up correctly get
              EXACTLY zero gradient still (gated multiplicatively by
              gate=0) -- that part is NOT a bug, don't "fix" it away.
      step 2 (after gate has moved off exactly 0 via one optimizer step):
              down/dw/up now get nonzero gradient too, since their gradient
              is gated by gate's now-nonzero value."""
    print("=== Check 4: Convpass staggered wake-up (gate step1 -> down/dw/up step2), dead-branch regression guard ===")
    torch.manual_seed(1)
    model = build_student(in_channels=1, teachers=TEACHERS, convpass=True)
    model.train()
    x = torch.randn(1, 1, 64, 64, 64)

    def grads_for_brats(m):
        out = {}
        for n, p in m.named_parameters():
            # gate scalar: "...convpass_gate.brats" (dict key is the last component)
            # conv weights: "...convpass.brats.{down,dw,up}.weight" (dict key is a MIDDLE component)
            if n.endswith(".convpass_gate.brats") or ".convpass.brats." in n:
                out[n] = 0.0 if p.grad is None else p.grad.abs().sum().item()
        return out

    # Step 1
    feats = model.forward_with_features(x, "brats")
    loss = sum(f.pow(2).mean() for f in feats.values())
    loss.backward()
    g1 = grads_for_brats(model)
    gate_keys = [n for n in g1 if n.endswith(".convpass_gate.brats")]
    conv_keys = [n for n in g1 if n not in gate_keys]
    dead_gate_step1 = [n for n in gate_keys if g1[n] == 0.0]
    assert not dead_gate_step1, f"gate got ZERO gradient on step 1 (should be nonzero): {dead_gate_step1}"
    nonzero_conv_step1 = [n for n in conv_keys if g1[n] != 0.0]
    assert not nonzero_conv_step1, (
        f"down/dw/up got NONZERO gradient on step 1 -- expected exactly zero at this point "
        f"(gated by gate=0); got nonzero for: {nonzero_conv_step1} -- unexpected, re-check the math."
    )
    print(f"  step 1: gate nonzero ({len(gate_keys)} params) OK, down/dw/up exactly zero "
          f"({len(conv_keys)} params) OK (expected -- gated by gate=0).")

    # Manually apply one SGD-like update to gate params only (stand-in for
    # optimizer.step(); avoids depending on run_pretrain.py's optimizer here).
    with torch.no_grad():
        for stage in model.decoder.stages:
            if stage.convpass_enabled:
                stage.convpass_gate["brats"] -= 0.1 * stage.convpass_gate["brats"].grad
    model.zero_grad()

    # Step 2
    feats = model.forward_with_features(x, "brats")
    loss = sum(f.pow(2).mean() for f in feats.values())
    loss.backward()
    g2 = grads_for_brats(model)
    dead_conv_step2 = [n for n in conv_keys if g2[n] == 0.0]
    assert not dead_conv_step2, (
        f"DEAD GRADIENT regression: down/dw/up still EXACTLY zero on step 2 after gate moved off 0: "
        f"{dead_conv_step2}. Check `up`'s init is Kaiming, not zeros_."
    )
    print(f"  step 2 (after gate moved off 0): down/dw/up now nonzero ({len(conv_keys)} params) OK.\n")


def check_param_counts(ver3):
    print("=== Param count sanity (vs ver3 spec's hand-computed ~37K/teacher) ===")
    gate_params = ver3.convpass_gate_parameters()
    n_gate = sum(p.numel() for p in gate_params)
    n_convpass_conv = sum(p.numel() for n, p in ver3.named_parameters() if ".convpass." in n and n not in
                          [k for k, _ in ver3.named_parameters() if ".convpass_gate." in k])
    print(f"  gate+alpha scalars total: {n_gate} (expect 5 stages x 2 (alpha,gate) x {len(TEACHERS)} teachers "
          f"= {5*2*len(TEACHERS)})")
    print(f"  Convpass conv weights total (all teachers): {n_convpass_conv}")
    print(f"  Convpass conv weights per teacher: ~{n_convpass_conv // len(TEACHERS)} "
          f"(spec's hand-calc: ~37,116)\n")


if __name__ == "__main__":
    ver3 = check_bit_identical()
    check_teacher_switch_changes_output(ver3)
    check_gradient_isolation(ver3)
    check_convpass_gradient_not_dead()
    check_param_counts(ver3)
    print("=== ALL STEP 1 MOCK CHECKS PASSED ===")
