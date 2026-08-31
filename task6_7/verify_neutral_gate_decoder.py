import sys, copy
sys.path.insert(0, "/root/FOMO26")
import torch
from networks.student import StudentResEncUNet

torch.manual_seed(0)
CKPT = "/root/FOMO26/expr/pretraining/ver5/checkpoints/step_280000.pt"
raw = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = {k[len("model."):]: v for k, v in raw["state_dict"].items() if k.startswith("model.")}
teachers = sorted({k.split(".convpass.")[1].split(".")[0] for k in sd if ".convpass." in k})

model = StudentResEncUNet(in_channels=1, teachers=teachers, convpass=True,
                           convpass_encoder=True, skip_alpha=False, norm_conditional=False)
missing, unexpected = model.load_state_dict(sd, strict=False)
assert len(missing) == 0

teacher_name = "brats"
dec_stage = model.decoder.stages[0]
in_ch = dec_stage.upsample.in_channels
skip_ch = dec_stage.block.conv1.in_channels - dec_stage.upsample.out_channels

x_dec = torch.randn(1, in_ch, 2, 2, 2)
skip_dec = torch.randn(1, skip_ch, 4, 4, 4)

model.eval()
with torch.no_grad():
    real_dec = dec_stage(x_dec, skip_dec, teacher_name)
    zeroed_dec_stage = copy.deepcopy(dec_stage)
    zeroed_dec_stage.convpass_gate[teacher_name].fill_(0.0)
    zeroed_dec = zeroed_dec_stage(x_dec, skip_dec, teacher_name)

    x_up = dec_stage.upsample(x_dec)
    cat = torch.cat([x_up, skip_dec], dim=1)
    block_only_dec = dec_stage.block(cat, teacher_name)

print("real gate value:", float(dec_stage.convpass_gate[teacher_name]))
print("real vs block-only max abs diff:", (real_dec - block_only_dec).abs().max().item())
dec_equal = torch.equal(zeroed_dec, block_only_dec)
print(f"decoder: zeroed-gate vs block-only bit-identical: {dec_equal}")
assert dec_equal, "FAIL"
print("PASS: decoder neutral path confirmed bit-identical too.")
