"""Experimental static schedules must preserve state and runtime active lengths."""
import pytest
from monolith.runtime import _native as nt
from tools.bench.gdn_static_bench import build, checked_run, snapshots


@pytest.mark.parametrize('kind,sl,workers,sgs', [
    ('head',8,0,16), ('static_head',8,5,16),
    ('static_stage',4,4,12), ('static_stage',2,4,8),
])
@pytest.mark.parametrize('hk,hv', [(8,16),(16,48)])
def test_static_schedule_continuation(kind,sl,workers,sgs,hk,hv):
    dev=nt.Device(); q=nt.Queue(dev)
    vs=[build(dev,c,hk=hk,hv=hv) for c in [('baseline',2,0,32),(kind,sl,workers,sgs)]]
    for step,(active,done) in enumerate([(8,0),(3,0),(1,0),(0,0),(6,0),(8,1),(8,0)]):
        for v in vs:
            v['st'].write(v['layout'].pack({'step':step,'t_this_step':active,'done':done}),0)
            before=snapshots(v)
            checked_run(q,v)
            if done or active==0:
                assert snapshots(v)==before
        assert snapshots(vs[0])==snapshots(vs[1])
    # GPU replay uses persistent monotone barrier epochs; no CPU counter reset.
    for v in vs:
        checked_run(q,v,repeat=64)
    assert snapshots(vs[0])==snapshots(vs[1])
