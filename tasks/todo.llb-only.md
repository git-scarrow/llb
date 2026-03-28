# tasks/todo.llb-only.md

## Now
- [ ] Implement LB config changes: AUTO_MIN_NODES=2, model classes,C, default stickiness off-lb-policy P1 M #llb #lb-policy  <!-- id:implement-lb-config-changes-autominnodes2-model-classesc-default -->
- [ ] Add cross-model fallback with x-fallback-model header-lb-policy P1 M #llb #lb-policy  <!-- id:add-cross-model-fallback-with-x-fallback-model-header-lb-policy -->
- [ ] Implement hedging for small models: t_hedge=0.6*p95 capped 800ms-lb P1 M #llb #latency  <!-- id:implement-hedging-for-small-models-thedge06p95-capped-800ms-lb -->
- [ ] Eligibility gate: require healthy, p95<threshold; integrate circuit breaker-lb P1 L #llb #eligibility  <!-- id:eligibility-gate-require-healthy-p95threshold-integrate-circuit -->
- [ ] Observability: add response headers and streaming events-lb P2 S #llb #observability  <!-- id:observability-add-response-headers-and-streaming-events-lb -->
- [ ] Verification: eligible_nodes check + load test + metrics review-lb P2 M #llb #verify  <!-- id:verification-eligiblenodes-check-load-test-metrics-review-lb -->
- [ ] Docs: update README with LB policy and ops steps-lb P3 S #llb #docs  <!-- id:docs-update-readme-with-lb-policy-and-ops-steps-lb -->

## Next

## Later

## Done

