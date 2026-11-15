# free-model-aggregator-01

One server, two wires, ten free providers. It reads its brief
(`MAKEFUN_PROMPT_FILE`), probes the ten providers' catalog routes, registers
with the trial env (`MAKEFUN_ENV_URL`), and answers the client's OpenAI Chat
Completions and Anthropic Messages requests — JSON and SSE — by routing each
one to the strongest free model that meets the request's requirements and
still has budget.

## How it routes

- Candidates are the free models whose capability flags cover the request
  (tools / vision / JSON / minimum context), ranked by the brief's rating
  table. Ties break toward the provider with the fewest calls so far, so
  load spreads instead of one provider being drained first.
- The request's requirements come from the workload manifest, keyed by the
  request id the env reports in-flight on `GET /work` (the client sends the
  manifest in order, one request at a time); the manifest order and the
  request body itself are fallbacks.
- A refused provider (402/429/5xx) is counted and the request is answered
  by the next-best model. Each candidate is tried twice: a refusal may be
  the provider's own one-off failure schedule or its budget running out —
  the two look identical, so a retry rescues the scheduled fail at the best
  rating, and a retry on an exhausted provider only costs one extra
  upstream call (the tie-break), never a budget unit that could have served
  later, because requests arrive in weight order.
- An OpenAI client request in JSON mode only rides a wire that can carry
  `response_format` (openai or gemini); the anthropic wire always echoes
  `json_mode: false`. Provider calls are always non-streaming: the full
  answer (text + usage + stop family) is fetched once, then rendered in the
  wire the client asked for, streamed or not. That keeps the echo exact —
  system text, last user text, tool names in order, max_tokens, stops,
  image bytes and JSON mode are carried through untouched on every wire.
- At most one inference call is ever in flight (a global gate), so the
  one-call-at-a-time provider is never double-booked. No wall clock and no
  randomness anywhere in an answer: fixed model-list timestamps,
  deterministic response ids.

## Run

```sh
node main.ts --list   # print the catalog, exit 0
node main.ts          # serve: register, answer 20 requests, exit 0
```

Node 22, standard library only. No model calls of its own, no keys, no
purchases — the trial's providers are free and their keys come from the
brief.

## Local practice scores (kit harness, `program/env/server.mjs` + `program/scorer/score.mjs`)

| trial | score | quality | served | wire | failover | catalog | upstream calls |
|-------|------:|--------:|-------:|-----:|---------:|--------:|---------------:|
| 0     | 99    | 99      | 100    | 100  | 100      | 100     | 39             |
| 1     | 99    | 99      | 100    | 100  | 100      | 100     | 47             |
| 2     | 99    | 99      | 100    | 100  | 100      | 100     | 47             |

20 of 20 requests valid on every trial; 233/233 client checks passed on
trial-0. Quality 99 is the ceiling on trial-0: the top model only has 10
successful calls at the best rating for 17 requests that want it, so the
remaining requests necessarily fall to the next-best rating — the routing
reaches the theoretical maximum earned rating (38213/38248).

## License

MIT. See `LICENSE`.
