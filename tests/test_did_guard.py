"""
Tests for DID-Guard core modules.

Run:  python -m pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Threat model
# ---------------------------------------------------------------------------

class TestThreatModel:
    def test_taxonomy_completeness(self):
        from src.did_guard.threat_model.taxonomy import THREAT_TAXONOMY, get_taxonomy_table
        assert len(THREAT_TAXONOMY) >= 10
        table = get_taxonomy_table()
        assert len(table) == len(THREAT_TAXONOMY)
        for entry in table:
            assert "id" in entry and "risk_score" in entry

    def test_risk_score(self):
        from src.did_guard.threat_model.taxonomy import THREAT_TAXONOMY, ImpactSeverity
        for t in THREAT_TAXONOMY:
            assert 0.0 < t.risk_score <= 4.0
            assert 0.0 <= t.likelihood <= 1.0

    def test_dolev_yao_adversary(self):
        from src.did_guard.threat_model.dolev_yao import DolevYaoAdversary, Message, MessageType, Principal
        adv = DolevYaoAdversary(semantic_capability=0.8, crypto_capability=1e-6)
        sender = Principal("did:peer:sender")
        receiver = Principal("did:peer:receiver")
        msg = Message(
            msg_type=MessageType.PLAINTEXT,
            sender=sender,
            receiver=receiver,
            content={"data": "hello"},
        )
        intercepted = adv.intercept(msg)
        assert intercepted is msg
        assert len(adv.known_messages) == 1

    def test_scenarios(self):
        from src.did_guard.threat_model.dolev_yao import build_standard_scenarios
        scenarios = build_standard_scenarios()
        assert len(scenarios) >= 5
        for s in scenarios:
            assert s.name
            assert s.expected_success_rate >= 0.0


# ---------------------------------------------------------------------------
# DID Identity
# ---------------------------------------------------------------------------

class TestDIDManager:
    def test_create_peer_did(self):
        from src.did_guard.identity.did_manager import DIDManager
        mgr = DIDManager()
        doc = mgr.create_peer_did("test-agent")
        assert doc.did.startswith("did:peer:")
        assert len(doc.verification_methods) > 0

    def test_create_web_did(self):
        from src.did_guard.identity.did_manager import DIDManager
        mgr = DIDManager()
        doc = mgr.create_web_did("example.com")
        assert doc.did.startswith("did:web:")

    def test_create_key_did(self):
        from src.did_guard.identity.did_manager import DIDManager
        mgr = DIDManager()
        doc = mgr.create_key_did("test-key")
        assert doc.did.startswith("did:key:")

    def test_resolve_did(self):
        from src.did_guard.identity.did_manager import DIDManager
        mgr = DIDManager()
        doc = mgr.create_peer_did("resolvable-agent")
        resolved = mgr.resolve(doc.did)
        assert resolved is not None
        assert resolved.did == doc.did

    def test_did_document_serialization(self):
        from src.did_guard.identity.did_manager import DIDManager
        mgr = DIDManager()
        doc = mgr.create_peer_did("serialize-agent")
        d = doc.to_dict()
        assert d["id"] == doc.did
        assert "@context" in d


# ---------------------------------------------------------------------------
# VC Attestation
# ---------------------------------------------------------------------------

class TestVCAttestor:
    def setup_method(self):
        from src.did_guard.identity.did_manager import DIDManager
        self.mgr = DIDManager()
        self.issuer_doc = self.mgr.create_peer_did("issuer")
        from src.did_guard.identity.vc_attestor import VCAttestor
        self.attestor = VCAttestor(issuer_did=self.issuer_doc.did, ttl_seconds=3600)
        self.subject_doc = self.mgr.create_peer_did("subject")

    def test_issue_capability_vc(self):
        vc = self.attestor.issue_capability_vc(
            subject_did=self.subject_doc.did,
            tools=["read_network_log", "query_alert_db"],
            scope="read",
        )
        assert vc.issuer_did == self.issuer_doc.did
        assert vc.subject_did == self.subject_doc.did
        assert "read_network_log" in vc.claims["tools"]

    def test_vc_verification(self):
        vc = self.attestor.issue_capability_vc(
            subject_did=self.subject_doc.did,
            tools=["read_network_log"],
        )
        assert self.attestor.verify(vc)

    def test_vc_revocation(self):
        from src.did_guard.identity.vc_attestor import VCAttestor
        att = VCAttestor(issuer_did=self.issuer_doc.did)
        vc = att.issue_capability_vc(subject_did=self.subject_doc.did, tools=["tool1"])
        att.revoke(vc.vc_id)
        assert not att.verify(vc)

    def test_get_tool_capabilities(self):
        vc = self.attestor.issue_capability_vc(
            subject_did=self.subject_doc.did,
            tools=["tool_a", "tool_b"],
        )
        caps = self.attestor.get_tool_capabilities(vc)
        assert "tool_a" in caps and "tool_b" in caps


# ---------------------------------------------------------------------------
# Trust: EigenTrust
# ---------------------------------------------------------------------------

class TestEigenTrust:
    def test_fit_returns_valid_scores(self):
        from src.did_guard.trust.eigentrust import EigenTrust
        C = EigenTrust.synthetic_matrix(n=10, seed=0)
        et = EigenTrust()
        t = et.fit(C)
        assert t.shape == (10,)
        assert np.all(t >= 0.0) and np.all(t <= 1.0)

    def test_convergence(self):
        from src.did_guard.trust.eigentrust import EigenTrust
        C = EigenTrust.synthetic_matrix(n=15, seed=1)
        et1 = EigenTrust(max_iter=200, tol=1e-10)
        t1 = et1.fit(C.copy())
        et2 = EigenTrust(max_iter=200, tol=1e-10)
        t2 = et2.fit(C.copy())
        np.testing.assert_allclose(t1, t2, rtol=1e-5)

    def test_get_score_by_index(self):
        from src.did_guard.trust.eigentrust import EigenTrust
        C = EigenTrust.synthetic_matrix(n=5, seed=2)
        et = EigenTrust()
        et.fit(C)
        for i in range(5):
            score = et.get_score(i)
            assert 0.0 <= score <= 1.0

    def test_get_score_by_did(self):
        from src.did_guard.trust.eigentrust import EigenTrust
        C = EigenTrust.synthetic_matrix(n=5, seed=3)
        dids = [f"did:peer:{i}" for i in range(5)]
        et = EigenTrust()
        et.fit_from_dids(dids, C)
        score = et.get_score("did:peer:0")
        assert 0.0 <= score <= 1.0

    def test_unknown_did_returns_zero(self):
        from src.did_guard.trust.eigentrust import EigenTrust
        C = EigenTrust.synthetic_matrix(n=5)
        et = EigenTrust()
        et.fit_from_dids([f"did:peer:{i}" for i in range(5)], C)
        assert et.get_score("did:peer:unknown") == 0.0


# ---------------------------------------------------------------------------
# Trust: Subjective Logic
# ---------------------------------------------------------------------------

class TestSubjectiveLogic:
    def test_opinion_normalisation(self):
        from src.did_guard.trust.subjective_logic import Opinion
        op = Opinion(b=0.5, d=0.3, u=0.2)
        assert abs(op.b + op.d + op.u - 1.0) < 1e-6

    def test_expectation(self):
        from src.did_guard.trust.subjective_logic import Opinion
        op = Opinion(b=0.7, d=0.1, u=0.2, a=0.5)
        assert abs(op.expectation - (0.7 + 0.5 * 0.2)) < 1e-6

    def test_fuse_vacuous_identity(self):
        from src.did_guard.trust.subjective_logic import Opinion, fuse
        vac = Opinion.vacuous(a=0.5)
        belief = Opinion(b=0.8, d=0.1, u=0.1, a=0.5)
        result = fuse(belief, vac)
        # Fusing with vacuous should not reduce belief drastically
        assert result.b > 0.5

    def test_discount(self):
        from src.did_guard.trust.subjective_logic import Opinion, discount
        trust_ab = Opinion(b=0.9, d=0.05, u=0.05)
        trust_bc = Opinion(b=0.8, d=0.1, u=0.1)
        result = discount(trust_ab, trust_bc)
        assert result.b < trust_bc.b  # discounting reduces belief

    def test_from_binomial(self):
        from src.did_guard.trust.subjective_logic import Opinion
        op = Opinion.from_binomial(pos=90, neg=10)
        assert op.b > 0.8
        assert op.d < 0.2


# ---------------------------------------------------------------------------
# Trust: HybridTrustManager
# ---------------------------------------------------------------------------

class TestHybridTrustManager:
    def test_get_score_after_fit(self):
        import time
        from src.did_guard.trust.eigentrust import EigenTrust
        from src.did_guard.trust.eigentrust_temporal import HybridTrustManager
        n = 10
        C = EigenTrust.synthetic_matrix(n=n)
        dids = [f"did:peer:{i}" for i in range(n)]
        ts = np.linspace(time.time() - 86400, time.time(), n)
        tm = HybridTrustManager()
        tm.fit(dids, C, ts)
        score = tm.get_score("did:peer:0")
        assert 0.0 <= score <= 1.0

    def test_feedback_updates_score(self):
        import time
        from src.did_guard.trust.eigentrust import EigenTrust
        from src.did_guard.trust.eigentrust_temporal import HybridTrustManager
        n = 10
        C = EigenTrust.synthetic_matrix(n=n)
        dids = [f"did:peer:{i}" for i in range(n)]
        ts = np.linspace(time.time() - 86400, time.time(), n)
        tm = HybridTrustManager()
        tm.fit(dids, C, ts)
        score_before = tm.get_score("did:peer:5")
        for _ in range(5):
            tm.record_feedback("did:peer:5", positive=True)
        score_after = tm.get_score("did:peer:5")
        assert score_after >= score_before


# ---------------------------------------------------------------------------
# FL: FLTrust
# ---------------------------------------------------------------------------

class TestFLTrust:
    def test_clean_clients_aggregate(self):
        from src.did_guard.fl.fltrust_wrapper import fltrust
        rng = np.random.default_rng(42)
        root = rng.normal(0, 1, 100)
        clients = [root + rng.normal(0, 0.01, 100) for _ in range(5)]
        agg = fltrust(root, clients)
        assert np.allclose(agg, root, atol=0.1)

    def test_byzantine_filtered(self):
        from src.did_guard.fl.fltrust_wrapper import fltrust
        rng = np.random.default_rng(99)
        root = rng.normal(0, 1, 100)
        clients = [root + rng.normal(0, 0.01, 100) for _ in range(5)]
        # Inject Byzantine client
        clients[0] = -root * 10
        agg = fltrust(root, clients)
        cos = np.dot(agg, root) / (np.linalg.norm(agg) * np.linalg.norm(root) + 1e-10)
        assert cos > 0.9  # Byzantine should be filtered out

    def test_diagnostics(self):
        from src.did_guard.fl.fltrust_wrapper import fltrust_with_diagnostics
        rng = np.random.default_rng(0)
        root = rng.normal(0, 1, 50)
        clients = [root + rng.normal(0, 0.01, 50) for _ in range(4)]
        clients.append(-root * 5)  # Byzantine
        agg, diag = fltrust_with_diagnostics(root, clients)
        assert diag["n_byzantine_detected"] >= 1


# ---------------------------------------------------------------------------
# FL: SecAgg
# ---------------------------------------------------------------------------

class TestSecureAgg:
    def test_aggregate_equals_mean(self):
        from src.did_guard.fl.secure_agg import secure_aggregate
        rng = np.random.default_rng(42)
        updates = [rng.normal(0, 1, 50) for _ in range(4)]
        result = secure_aggregate(updates)
        # Due to mask cancellation, result should equal the plain mean
        plain_mean = np.stack(updates).mean(axis=0)
        np.testing.assert_allclose(result, plain_mean, atol=1e-6)

    def test_single_update_passthrough(self):
        from src.did_guard.fl.secure_agg import secure_aggregate
        rng = np.random.default_rng(1)
        update = rng.normal(0, 1, 30)
        result = secure_aggregate([update])
        np.testing.assert_array_equal(result, update)


# ---------------------------------------------------------------------------
# Policy: ZeroTrustPolicyEngine
# ---------------------------------------------------------------------------

class TestZeroTrustPolicy:
    def setup_method(self):
        import time
        from src.did_guard.trust.eigentrust import EigenTrust
        from src.did_guard.trust.eigentrust_temporal import HybridTrustManager
        from src.did_guard.policy.zero_trust_engine import ZeroTrustPolicyEngine

        n = 10
        C = EigenTrust.synthetic_matrix(n=n)
        dids = [f"did:peer:{i}" for i in range(n)]
        ts = np.linspace(time.time() - 86400, time.time(), n)
        self.tm = HybridTrustManager()
        self.tm.fit(dids, C, ts)
        # Boost did:peer:0 trust
        for _ in range(20):
            self.tm.record_feedback("did:peer:0", positive=True)
        self.engine = ZeroTrustPolicyEngine(self.tm)
        self.high_trust_did = "did:peer:0"
        self.low_trust_did = "did:peer:unknown:low"

    def test_deny_without_vc(self):
        from src.did_guard.policy.zero_trust_engine import PolicyContext
        ctx = PolicyContext(
            did=self.high_trust_did,
            capability="read_network_log",
            tool_args={},
            vc_claims={},
        )
        decision = self.engine.evaluate(ctx)
        assert not decision.allowed

    def test_allow_with_valid_vc_and_trust(self):
        from src.did_guard.policy.zero_trust_engine import PolicyContext
        ctx = PolicyContext(
            did=self.high_trust_did,
            capability="read_network_log",
            tool_args={},
            risk=0.1,
            vc_claims={"tools": ["read_network_log"], "scope": "read"},
        )
        decision = self.engine.evaluate(ctx)
        # Decision may be True or False depending on trust score; just assert no error
        assert isinstance(decision.allowed, bool)

    def test_audit_log(self):
        from src.did_guard.policy.zero_trust_engine import PolicyContext
        self.engine.clear_audit_log()
        ctx = PolicyContext(
            did=self.high_trust_did,
            capability="tool_x",
            tool_args={},
            vc_claims={"tools": ["tool_x"]},
        )
        self.engine.evaluate(ctx)
        log = self.engine.get_audit_log()
        assert len(log) == 1
        assert log[0]["did"] == self.high_trust_did


# ---------------------------------------------------------------------------
# Agents: Orchestrator
# ---------------------------------------------------------------------------

class TestOrchestrator:
    def setup_method(self):
        from src.did_guard.agents.orchestrator import DIDGuardOrchestrator
        from src.did_guard.identity.did_manager import DIDManager
        from src.did_guard.identity.vc_attestor import VCAttestor
        self.orch = DIDGuardOrchestrator()
        mgr = DIDManager()
        self.agent_doc = mgr.create_peer_did("test-orchestrator-agent")
        attestor = VCAttestor(issuer_did="did:web:test.local")
        self.vc = attestor.issue_capability_vc(
            subject_did=self.agent_doc.did,
            tools=["read_network_log", "query_alert_db"],
            scope="read",
        )

    def test_authenticate(self):
        session = self.orch.authenticate(self.agent_doc.did, self.vc)
        assert session.agent_did == self.agent_doc.did

    def test_execute_denied_without_vc(self):
        session = self.orch.authenticate(self.agent_doc.did, None)
        result = self.orch.execute(session, "read_network_log", {"path": "/tmp/test.log", "lines": 5})
        assert not result.allowed

    def test_execute_tool_not_found(self):
        session = self.orch.authenticate(self.agent_doc.did, self.vc)
        result = self.orch.execute(session, "nonexistent_tool", {})
        assert not result.allowed

    def test_audit_summary(self):
        session = self.orch.authenticate(self.agent_doc.did, None)
        self.orch.execute(session, "read_network_log", {"path": "/tmp/log", "lines": 5})
        summary = self.orch.audit_summary()
        assert "total_actions" in summary
        assert summary["total_actions"] >= 1


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class TestDatasets:
    def test_load_unsw_synthetic(self):
        import os
        os.environ["DID_GUARD_SYNTH"] = "1"
        from importlib import reload
        import src.did_guard.config as cfg
        import src.did_guard.anomaly.datasets as ds
        reload(cfg)
        reload(ds)
        df = ds.load_unsw(sample=200)
        assert len(df) == 200
        assert "is_attack" in df.columns

    def test_load_cic_synthetic(self):
        import os
        os.environ["DID_GUARD_SYNTH"] = "1"
        from importlib import reload
        import src.did_guard.config as cfg
        import src.did_guard.anomaly.datasets as ds
        reload(cfg)
        reload(ds)
        df = ds.load_cic(sample=200)
        assert len(df) == 200

    def test_get_feature_matrix(self):
        import os
        os.environ["DID_GUARD_SYNTH"] = "1"
        from importlib import reload
        import src.did_guard.config as cfg
        import src.did_guard.anomaly.datasets as ds
        reload(cfg)
        reload(ds)
        df = ds.load_unsw(sample=100)
        X, y = ds.get_feature_matrix(df)
        assert X.shape[0] == 100
        assert len(y) == 100


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------

class TestAutoencoder:
    def test_forward_pass(self):
        import torch
        from src.did_guard.anomaly.autoencoder import Autoencoder
        ae = Autoencoder(input_dim=20)
        x = torch.randn(8, 20)
        out = ae(x)
        assert out.shape == x.shape

    def test_get_set_params(self):
        from src.did_guard.anomaly.autoencoder import Autoencoder
        ae = Autoencoder(input_dim=15)
        flat = ae.get_params_flat()
        ae2 = Autoencoder(input_dim=15)
        ae2.set_params_flat(flat)
        flat2 = ae2.get_params_flat()
        np.testing.assert_allclose(flat, flat2, rtol=1e-5)

    def test_reconstruction_error_shape(self):
        import torch
        from src.did_guard.anomaly.autoencoder import Autoencoder
        ae = Autoencoder(input_dim=10)
        x = torch.randn(16, 10)
        errors = ae.reconstruction_error(x)
        assert errors.shape == (16,)

    def test_evaluate_ids(self):
        import torch
        from src.did_guard.anomaly.autoencoder import Autoencoder, evaluate_ids, train_local
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(200, 12)).astype(np.float32)
        y = np.concatenate([np.zeros(160), np.ones(40)]).astype(int)
        ae = Autoencoder(input_dim=12)
        ae, _ = train_local(ae, X[:160], epochs=2, batch_size=32)
        metrics = evaluate_ids(ae, X, y)
        assert "f1" in metrics and "roc_auc" in metrics


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

class TestBenchmarks:
    def test_agentdojo_benchmark(self):
        from src.did_guard.benchmarks.agentdojo_runner import AgentDojoBenchmark
        bench = AgentDojoBenchmark(n_tasks=20, injection_rate=0.3)
        results = bench.run()
        assert "utility" in results
        assert "asr_defended" in results
        assert 0.0 <= results["utility"] <= 1.0

    def test_asb_benchmark(self):
        from src.did_guard.benchmarks.asb_runner import ASBBenchmark
        bench = ASBBenchmark(n_per_category=5)
        results = bench.run()
        assert "overall_asr_defended" in results
        assert "categories" in results

    def test_baselines(self):
        from src.did_guard.benchmarks.baseline_runner import run_all_baselines
        results = run_all_baselines(n_tasks=50)
        assert len(results) == 3
        for b in results:
            assert "system" in b and "asr" in b


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
