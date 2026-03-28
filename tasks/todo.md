# tasks/todo.md

_Exported: 2025-09-12 19:03:04_

## Now
- [>] Implement LB config changes: AUTO_MIN_NODES=2, model classes, default stickiness off P1 M #llb #lb-policy  <!-- id:implement-lb-config-changes-autominnodes2-model-classes-default -->
- [ ] Add cross-model fallback with x-fallback-model header P1 M #llb #lb-policy  <!-- id:add-cross-model-fallback-with-x-fallback-model-header -->
- [ ] Implement hedging for small models: t_hedge=0.6*p95 capped 800ms P1 M #llb #latency  <!-- id:implement-hedging-for-small-models-thedge06p95-capped-800ms -->
- [ ] Eligibility gate: require healthy, p95<threshold; integrate circuit breaker P1 L #llb #eligibility  <!-- id:eligibility-gate-require-healthy-p95threshold-integrate-circuit -->

## Next
- [ ] Observability: add response headers and streaming events P2 S #llb #observability  <!-- id:observability-add-response-headers-and-streaming-events -->
- [ ] Verification: eligible_nodes check + load test + metrics review P2 M #llb #verify  <!-- id:verification-eligiblenodes-check-load-test-metrics-review -->

## Later
- [ ] Docs: update README with LB policy and ops steps P3 S #llb #docs  <!-- id:docs-update-readme-with-lb-policy-and-ops-steps -->

## Done

---
Conventions: '- [ ] task P1 S #tag due:2025-09-30' | in-progress shown as [>]
