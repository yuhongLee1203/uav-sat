#!/usr/bin/env python3
"""Record Weighted Average and Mean Shift from identical 6x6 regularized logits."""
import sys
from pathlib import Path

path = Path(sys.argv[1])
source = path.read_text()
anchor = '            regularized_ms_logits = tau * combined_log_probability\n'
insert = '''            regularized_ms_logits = tau * combined_log_probability
            if ms_centers.shape[1] != 36:
                raise RuntimeError("decoder probe requires the original 6x6/36 final candidates")
            decoder_weights = torch.softmax(
                regularized_ms_logits / max(tau, 1e-6), dim=1
            )
            decoder_weighted_xy = (
                decoder_weights.unsqueeze(-1) * ms_centers
            ).sum(dim=1)[0]
'''
assert source.count(anchor) == 1, 'final logits anchor changed'
source = source.replace(anchor, insert)
anchor = '                "ms_x": float(ms_xy[0]),\n'
insert = '''                "decoder_weighted_x": float(decoder_weighted_xy[0].item()),
                "decoder_weighted_y": float(decoder_weighted_xy[1].item()),
                "decoder_candidate_centers_json": json.dumps(ms_centers[0].detach().cpu().tolist()),
                "decoder_candidate_logits_json": json.dumps(regularized_ms_logits[0].detach().cpu().tolist()),
                "ms_x": float(ms_xy[0]),
'''
assert source.count(anchor) == 1, 'frame row anchor changed'
source = source.replace(anchor, insert)
path.write_text(source)
