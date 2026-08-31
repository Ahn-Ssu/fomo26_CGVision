"""Verifies the Task 6/7 spec's §3.1 "neutral path" claim: with
convpass_gate[teacher]=0.0, EncoderStage/DecoderStage.forward collapses to
block-only output, bit-identical (torch.equal, not allclose) to skipping
Convpass entirely. Run once before any real embedding extraction.
"""
import sys
sys.path.insert(0, "/root/FOMO26")
import torch
from networks.student import StudentResEncUNet

torch.manual_seed(0)

CKPT = "/root/FOMO26/expr/pretraining/ver5/checkpoints/step_280000.pt"
raw = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = {k[len("model."):]: v for k, v in raw["state_dict"].items() if k.startswith("model.")}

teachers = sorted({k.split(".convpass.")[1].split(".")[0] for k in sd if ".convpass." in k})
print("teachers:", teachers)

model = StudentResEncUNet(in_channels=1, teachers=teachers, convpass=True,
                           convpass_encoder=True, skip_alpha=False, norm_conditional=False)
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"loaded: missing={len(missing)} unexpected={len(unexpected)} (expect near-0 missing for in_channels=1 exact match)")
assert len(missing) == 0, missing

x = torch.randn(1, 1, 32, 32, 32)  # small spatial size, CPU-fast; exactness doesn't depend on size
teacher_name = "brats"

model.eval()
with torch.no_grad():
    # (a) real trained forward with the selected teacher
    real_out = model.encoder.stages[0](x, teacher_name)

    # (b) same stage, gate manually zeroed (a deep-copied stage so the real
    # trained gate value is untouched)
    import copy
    zeroed_stage = copy.deepcopy(model.encoder.stages[0])
    with torch.no_grad():
        zeroed_stage.convpass_gate[teacher_name].fill_(0.0)
    zeroed_out = zeroed_stage(x, teacher_name)

    # (c) block-only output (manually bypassing the convpass addition entirely)
    block_only_out = model.encoder.stages[0].block(x, teacher_name)

print("real vs zeroed equal:      ", torch.equal(real_out, zeroed_out) if False else "N/A (real has nonzero gate, expected to differ)")
print("real gate value:", float(model.encoder.stages[0].convpass_gate[teacher_name]))
diff = (real_out - block_only_out).abs().max().item()
print(f"real vs block-only max abs diff: {diff:.6f} (expected > 0 -- gate is trained, nonzero)")

zeroed_vs_block_equal = torch.equal(zeroed_out, block_only_out)
print(f"zeroed-gate vs block-only bit-identical (torch.equal): {zeroed_vs_block_equal}")
assert zeroed_vs_block_equal, "FAIL: gate=0 forward is NOT bit-identical to block-only (neutral) forward"

# Also check a decoder stage (uses concatenated input + skip_alpha=False for ver5)
dec_stage = model.decoder.stages[0]
skip = torch.randn(1, dec_stage.block.conv1.in_channels - x.shape[1] if False else 320, 4, 4, 4)
# simpler: just re-derive proper skip channel count from the stage's own conv1 in_channels
in_ch = dec_stage.block.conv1.in_channels
x_dec = torch.randn(1, in_ch - 320, 4, 4, 4) if in_ch > 320 else torch.randn(1, in_ch, 4, 4, 4)
skip_dec = torch.randn(1, 320, 4, 4, 4)
with torch.no_grad():
    real_dec = dec_stage(x_dec, skip_dec, teacher_name)
    zeroed_dec_stage = copy.deepcopy(dec_stage)
    zeroed_dec_stage.convpass_gate[teacher_name].fill_(0.0)
    zeroed_dec = zeroed_dec_stage(x_dec, skip_dec, teacher_name)
    cat = torch.cat([x_dec, skip_dec], dim=1)
    block_only_dec = dec_stage.block(cat, teacher_name)

dec_equal = torch.equal(zeroed_dec, block_only_dec)
print(f"decoder: zeroed-gate vs block-only bit-identical: {dec_equal}")
assert dec_equal, "FAIL: decoder gate=0 forward is NOT bit-identical to block-only forward"

print("\nPASS: neutral path (gate=0) is bit-identical to Convpass-free (ver1-structure) forward, "
      "for both encoder and decoder stages.")
