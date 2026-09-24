"""CPU flow integration contracts; no claim of full-model GPU acceptance."""
import copy
from pathlib import Path
import dill
import hydra
from omegaconf import OmegaConf
import pytest
import torch

from test_p2n_new_policy import (META, DINO_CONFIG, PROCESSOR, local_mock_dino,
                                tokenizer_config, observation)
from oat.model.common.normalizer import SingleFieldLinearNormalizer
from oat.perception.dinov3_patch_encoder import DINOv3PatchEncoder
from oat.perception.latent_flow_token_obs_encoder import FlowTokenObservationEncoder
from oat.policy.p2n_latent_flow import P2NLatentFlowPolicy
from oat.policy.p2n_state_gate_latent_flow import P2NStateGateLatentFlowPolicy
from oat.policy.p2n_latent_flow_common import prepare_flow_training_batch


def make_flow(gate=False, *, queries=2, history_gate_mode='learned'):
    dino = DINOv3PatchEncoder(load_mode='restore', config=DINO_CONFIG,
        processor_config=PROCESSOR, revision='a'*40, weight_sha256='b'*64, brightness=.1)
    dino.load_state_dict(dino.state_dict(), strict=True)
    obs = FlowTokenObservationEncoder(META, n_obs_steps=2, n_emb=16, n_head=2,
        ffn_dim=32, num_queries=queries, resampler_depth=1, dino_encoder=dino)
    tc = tokenizer_config()
    for key in ('encoder','decoder'):
        tc[key].update(sample_horizon=16, latent_dim=5)
    tc['encoder']['num_registers'] = 8
    tc['decoder']['latent_horizon'] = 8
    tc['quantizer']['levels'] = [8,5,5,5,5]
    tok = hydra.utils.instantiate(tc)
    tok.normalizer['action'] = SingleFieldLinearNormalizer.create_identity()
    kwargs = dict(shape_meta=META, obs_encoder=obs, action_tokenizer=tok,
        tokenizer_config=tc, embed_dim=16, n_layers=2, n_heads=2, ffn_dim=32,
        dropout=0., activation_checkpointing=True, self_past_p=1.,
        self_past_warmup_steps=0, self_past_ramp_steps=0, self_past_chunk_size=2)
    if gate:
        kwargs.update(history_embed_dim=16, history_n_heads=2, history_n_layers=1,
            history_dropout=0., history_gate_hidden_dim=16, history_gate_mode=history_gate_mode)
    return (P2NStateGateLatentFlowPolicy if gate else P2NLatentFlowPolicy)(**kwargs)


def make_batch(policy, size=4):
    valid = torch.ones(size,7,dtype=torch.bool)
    result = dict(obs=observation(policy,size,valid), action=torch.randn(size,16,7)*.1,
        past_action=torch.randn(size,7,7)*.1, past_action_valid=valid,
        prev_obs=observation(policy,size,valid), prev_past_action=torch.randn(size,7,7)*.1,
        prev_past_action_valid=valid.clone(), prev_window_valid=torch.ones(size,dtype=torch.bool),
        future_action_valid=torch.ones(size,16,dtype=torch.bool), sample_id=torch.arange(size))
    result['future_action_valid'][-1,1:]=False
    return result


@pytest.fixture(autouse=True)
def limit_threads():
    previous=torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('gate',[False,True])
def test_packed_teacher_student_updates_and_layout(gate):
    torch.manual_seed(23)
    student=make_flow(gate,queries=64).train()
    teacher=copy.deepcopy(student).requires_grad_(False).eval()
    batch=make_batch(student)
    generator=torch.Generator().manual_seed(912)
    opt=student.get_optimizer(policy_lr=.005,obs_enc_lr=.005)
    frozen={n:p.detach().clone() for n,p in student.named_parameters() if not p.requires_grad}
    if not gate:
        assert not any('history_encoder' in n or 'history_gate' in n or 'observation_pool' in n for n,_ in student.named_parameters())
    assert not any('teacher' in n or 'token_embedding' in n for n in student.state_dict())
    for _ in range(3):
        prepared=prepare_flow_training_batch(batch,student=student,teacher=teacher,
            generator=generator,self_past_generator=generator)
        assert prepared.frozen_patches.grad_fn is None
        with torch.autocast('cpu',dtype=torch.bfloat16):
            loss=student(prepared)
        assert loss.dtype==torch.float32 and torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in student.parameters())
        opt.step(); opt.zero_grad(set_to_none=True)
        student.on_optimizer_step()
    context=student.build_context(batch['obs'],batch['past_action'],batch['past_action_valid'])
    assert context.memory.shape[1]==(271 if gate else 267)
    context.validate_variant(student.variant)
    assert student.self_past_step==3
    assert all(torch.equal(p,frozen[n]) for n,p in student.named_parameters() if n in frozen)
    assert all(p.grad is None for p in teacher.parameters())
    assert student._last_flow_losses.keys()=={'loss','fm_loss','ct_loss'}


@pytest.mark.parametrize('gate',[False,True])
def test_explicit_history_prediction_validation_and_acknowledgement(gate):
    policy=make_flow(gate).eval()
    batch=make_batch(policy,1)
    result=policy.predict_action(batch['obs'],past_actions=batch['past_action'],
        past_action_valid=batch['past_action_valid'],generator=torch.Generator().manual_seed(2))
    assert result['action'].shape==(1,8,7) and result['action_pred'].shape==(1,16,7)
    assert policy._past_buffer is None and policy._pending_execution_steps is None
    obs=policy.create_dummy_observation(1)
    result=policy.predict_action(obs,generator=torch.Generator().manual_seed(5))
    policy.record_executed_actions(result['action'],executed_lengths=[0])
    assert not policy._past_valid_buffer.any()
    result=policy.predict_action(obs)
    policy.record_executed_actions(result['action'],executed_lengths=[3])
    assert policy._past_valid_buffer.sum()==3
    before=(policy._past_buffer.clone(), policy._past_valid_buffer.clone())
    metrics=policy.validation_metrics(batch,torch.Generator().manual_seed(11),'generated')
    assert metrics['decoded_action_mse']['count']==7
    assert metrics['projection_legal']['sum']==metrics['projection_legal']['count']
    assert torch.equal(before[0],policy._past_buffer) and torch.equal(before[1],policy._past_valid_buffer)
    with pytest.raises((ValueError,TypeError)):
        policy.predict_action(batch['obs'],past_actions=batch['past_action'],past_action_valid=batch['past_action_valid'],topk=3)
    policy.reset(); assert policy._past_buffer is None


def test_closed_gate_has_no_global_history_bypass():
    policy=make_flow(True,history_gate_mode='closed').eval()
    # Move beyond zero-init so this is a real conditioning invariance check.
    for name,p in policy.model.named_parameters():
        if 'modulation' in name or 'output' in name:
            torch.nn.init.normal_(p,std=.05)
    batch=make_batch(policy,1)
    obs2=copy.deepcopy(batch['obs'])
    obs2['state_history__robot0_eef_pos'] += 50
    def predict(obs):
        return policy.predict_action(obs,past_actions=batch['past_action'],
            past_action_valid=batch['past_action_valid'],generator=torch.Generator().manual_seed(31))['action_pred']
    assert torch.equal(predict(batch['obs']),predict(obs2))
    assert torch.equal(policy.current_state_features(batch['obs']),policy.current_state_features(obs2))


@pytest.mark.parametrize('gate',[False,True])
def test_offline_artifact_strict_roundtrip_and_family_rejection(gate,tmp_path):
    policy=make_flow(gate).eval()
    metadata=policy.artifact_metadata()
    cfg=OmegaConf.create({'training':{'use_ema':True},'policy':policy.export_config()})
    payload=dict(cfg=cfg,policy_config=policy.export_config(),metadata=metadata,
                 state_dicts={'model':policy.state_dict(),'ema_model':policy.state_dict()})
    path=tmp_path/'flow.ckpt'; torch.save(payload,path,pickle_module=dill)
    restored=type(policy).from_checkpoint(path)
    assert restored.variant==policy.variant
    assert restored._past_buffer is None
    for key,value in policy.state_dict().items():
        assert torch.equal(value,restored.state_dict()[key])
    wrong=P2NLatentFlowPolicy if gate else P2NStateGateLatentFlowPolicy
    with pytest.raises(ValueError,match='family/variant'):
        wrong.from_checkpoint(path)
    payload['metadata']['policy_family']='autoregressive'
    torch.save(payload,path,pickle_module=dill)
    with pytest.raises(ValueError,match='family/variant'):
        type(policy).from_checkpoint(path)


def test_teacher_uses_own_adapter_and_one_shared_frozen_patch_realization(monkeypatch):
    from oat.model.flow.consistency_flow import FlowSchedule
    import oat.policy.p2n_latent_flow_common as common
    student=make_flow().train(); student.self_past_p=0
    teacher=copy.deepcopy(student).requires_grad_(False).eval()
    with torch.no_grad():
        teacher.obs_encoder.patch_projection.weight.add_(.1)
    batch=make_batch(student)
    fm=torch.tensor([0,1,2]); ct=torch.tensor([3])
    monkeypatch.setattr(common,'sample_flow_schedule',lambda *args,**kwargs:
        FlowSchedule(torch.full((4,),.2),torch.tensor([0.,0.,0.,.3]),fm,ct))
    calls={'student_backbone':0,'teacher_backbone':0,'student_adapter':[],'teacher_adapter':[]}
    def backbone(name):
        def record(*args): calls[name]+=1
        return record
    hooks=[student.obs_encoder.dino_encoder.backbone.register_forward_hook(backbone('student_backbone')),
           teacher.obs_encoder.dino_encoder.backbone.register_forward_hook(backbone('teacher_backbone')),
           student.obs_encoder.patch_projection.register_forward_pre_hook(lambda _,args:calls['student_adapter'].append(args[0].detach().clone())),
           teacher.obs_encoder.patch_projection.register_forward_pre_hook(lambda _,args:calls['teacher_adapter'].append(args[0].detach().clone()))]
    try:
        prepared=common.prepare_flow_training_batch(batch,student=student,teacher=teacher,
            generator=torch.Generator().manual_seed(83))
        assert calls['student_backbone']==1 and calls['teacher_backbone']==0
        assert len(calls['teacher_adapter'])==1 and not calls['student_adapter']
        assert torch.equal(calls['teacher_adapter'][0],prepared.frozen_patches[ct].flatten(0,2))
        student(prepared).backward()
        assert len(calls['student_adapter'])==1
        assert torch.equal(calls['student_adapter'][0],prepared.frozen_patches.flatten(0,2))
    finally:
        for hook in hooks: hook.remove()


@pytest.mark.parametrize('gate',[False,True])
def test_self_past_excludes_all_invalid_previous_windows(gate,monkeypatch):
    policy=make_flow(gate).train()
    batch=make_batch(policy)
    batch['prev_window_valid'].zero_()
    if gate:
        batch['prev_obs']['state_history_valid'].zero_()
    def forbidden(*args,**kwargs):
        raise AssertionError('No valid previous windows: generation must not execute')
    monkeypatch.setattr(policy,'_generate_actions',forbidden)
    past=policy._maybe_self_past(batch,batch['past_action'],probability=1.)
    assert torch.equal(past,batch['past_action']) and policy.training
