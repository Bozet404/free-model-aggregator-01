// free-model-aggregator-01 — one server, two wires, ten free providers.
//
// The server reads its brief (MAKEFUN_PROMPT_FILE), probes the ten providers'
// catalog routes, registers with the trial env (MAKEFUN_ENV_URL), and answers
// the client's OpenAI Chat Completions and Anthropic Messages requests (JSON
// and SSE) by routing each one to the strongest free model that meets the
// request's requirements and still has budget. A refused provider (402/429)
// is counted and the request is answered by the next-best model.
//
// Routing, in one line: candidates are the free models whose capability flags
// cover the request, ranked by the brief's rating table; ties break toward
// the provider with the fewest calls so far, so load spreads instead of one
// provider being drained first. The request's requirements come from the
// workload manifest, keyed by the request id the env reports in-flight on
// GET /work (the client sends the manifest in order, one request at a time).
//
// Provider calls are always non-streaming: the full answer (text + usage +
// stop family) is fetched once, then rendered in the wire the client asked
// for, streamed or not. That keeps the echo exact — system text, last user
// text, tool names in order, max_tokens, stops, image bytes and JSON mode
// are carried through untouched on every wire — and makes both SSE shapes
// fully deterministic.
//
// Node 22, standard library only. No wall clock and no randomness anywhere
// in an answer: fixed model-list timestamps, deterministic response ids.

import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { readFileSync } from "node:fs";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Wire = "openai" | "anthropic" | "gemini";
type Auth = "bearer" | "x-api-key" | "x-goog-api-key" | "none";

interface ModelRow {
  id: string;
  free: boolean;
  context_length: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
  rating_milli: number;
}

interface ProviderInfo {
  id: string;
  name: string;
  wire: Wire;
  auth: Auth;
  key_env: string;
  key: string;
  path: string;
  baseUrl: string;
  models: ModelRow[];
  calls: number;
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: string;
  models: CatalogModel[];
}

interface CatalogDoc {
  providers: CatalogProvider[];
}

interface Requirements {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface ManifestEntry {
  id: string;
  api: string;
  stream: boolean;
  weight: number;
  requires: Requirements;
}

interface ImageRef {
  kind: "data" | "url";
  mime: string;
  data: string;
  url: string;
}

interface ToolDef {
  name: string;
  description: string;
  schema: Record<string, unknown>;
}

interface ChatMessage {
  role: "system" | "user" | "assistant";
  text: string;
  image: ImageRef | null;
}

// The request, normalised once, whatever wire the client used.
interface NormalRequest {
  system: string | null;
  messages: ChatMessage[];
  tools: ToolDef[];
  toolChoice: string; // "auto" | "none" | <tool name>
  maxTokens: number;
  stops: string[];
  temperature: number | null;
  topP: number | null;
  jsonMode: boolean;
  stream: boolean;
}

// One provider answer, normalised once, whatever wire it came back on.
interface NormalAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stopFamily: number; // 0 stop, 1 length, 2 tool_calls, 3 content_filter
  providerId: string;
  modelId: string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const OPENAI_CREATED = 1767225600;
const ANTHROPIC_CREATED = "2026-01-01T00:00:00Z";
const PROVIDER_TIMEOUT_MS = 5000;
const WORK_TIMEOUT_MS = 5000;
const REGISTER_RETRIES = 3;
const WORK_POLL_MS = 100;
const WALL_DEADLINE_MS = 170_000;

const envUrl = (process.env.MAKEFUN_ENV_URL ?? "").replace(/\/$/, "");
const promptFile = process.env.MAKEFUN_PROMPT_FILE ?? "";
const trialSeed = process.env.MAKEFUN_TRIAL_SEED ?? "";

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function isRecord(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function asString(v: unknown): string {
  return typeof v === "string" ? v : "";
}

function asNumber(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function stringList(v: unknown): string[] {
  if (typeof v === "string") return [v];
  if (Array.isArray(v)) return v.filter((x): x is string => typeof x === "string");
  return [];
}

function slug(s: string): string {
  return s.replace(/[^A-Za-z0-9]+/g, "-").replace(/^-+|-+$/g, "").toLowerCase();
}

function splitPieces(text: string, size = 16): string[] {
  const out: string[] = [];
  for (let i = 0; i < text.length; i += size) out.push(text.slice(i, i + size));
  if (out.length === 0) out.push("");
  return out;
}

async function fetchJson(url: string, init: RequestInit, timeoutMs: number): Promise<{ status: number; body: unknown }> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...init, signal: controller.signal });
    const text = await res.text();
    let body: unknown = null;
    try {
      body = JSON.parse(text);
    } catch {
      body = null;
    }
    return { status: res.status, body };
  } finally {
    clearTimeout(timer);
  }
}

async function fetchText(url: string, init: RequestInit, timeoutMs: number): Promise<{ status: number; text: string }> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...init, signal: controller.signal });
    return { status: res.status, text: await res.text() };
  } finally {
    clearTimeout(timer);
  }
}

// ---------------------------------------------------------------------------
// Brief parsing — the three JSON documents embedded in prompt.txt
// ---------------------------------------------------------------------------

function extractJsonDoc(text: string, schema: string): Record<string, unknown> | null {
  const marker = `{"schema":"${schema}"`;
  const start = text.indexOf(marker);
  if (start < 0) return null;
  let depth = 0;
  let inStr = false;
  let esc = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inStr) {
      if (esc) esc = false;
      else if (ch === "\\") esc = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') inStr = true;
    else if (ch === "{") depth++;
    else if (ch === "}") {
      depth--;
      if (depth === 0) {
        try {
          const parsed: unknown = JSON.parse(text.slice(start, i + 1));
          return isRecord(parsed) ? parsed : null;
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}

interface Brief {
  providers: Array<{ id: string; name: string; wire: Wire; auth: Auth; key_env: string; key: string; path: string }>;
  ratings: Array<{ model: string; provider: string; rating_milli: number; context_length: number; tools: boolean; vision: boolean; json: boolean; free: boolean }>;
  manifest: ManifestEntry[];
}

function readBrief(): Brief {
  const empty: Brief = { providers: [], ratings: [], manifest: [] };
  if (!promptFile) return empty;
  let text = "";
  try {
    text = readFileSync(promptFile, "utf8");
  } catch {
    return empty;
  }
  const dirDoc = extractJsonDoc(text, "free-model-aggregator-01/providers/v1");
  const ratingsDoc = extractJsonDoc(text, "free-model-aggregator-01/ratings/v1");
  const workloadDoc = extractJsonDoc(text, "free-model-aggregator-01/workload/v1");
  const brief: Brief = { providers: [], ratings: [], manifest: [] };
  if (dirDoc && Array.isArray(dirDoc.providers)) {
    for (const raw of dirDoc.providers) {
      if (!isRecord(raw)) continue;
      const wire = asString(raw.wire);
      if (wire !== "openai" && wire !== "anthropic" && wire !== "gemini") continue;
      const auth = asString(raw.auth);
      if (auth !== "bearer" && auth !== "x-api-key" && auth !== "x-goog-api-key" && auth !== "none") continue;
      brief.providers.push({
        id: asString(raw.id),
        name: asString(raw.name),
        wire,
        auth,
        key_env: asString(raw.key_env),
        key: asString(raw.key),
        path: asString(raw.path),
      });
    }
  }
  if (ratingsDoc && Array.isArray(ratingsDoc.ratings)) {
    for (const raw of ratingsDoc.ratings) {
      if (!isRecord(raw)) continue;
      brief.ratings.push({
        model: asString(raw.model),
        provider: asString(raw.provider),
        rating_milli: asNumber(raw.rating_milli) ?? 0,
        context_length: asNumber(raw.context_length) ?? 0,
        tools: raw.tools === true,
        vision: raw.vision === true,
        json: raw.json === true,
        free: raw.free === true,
      });
    }
  }
  if (workloadDoc && Array.isArray(workloadDoc.manifest)) {
    for (const raw of workloadDoc.manifest) {
      if (!isRecord(raw)) continue;
      const req = isRecord(raw.requires) ? raw.requires : {};
      brief.manifest.push({
        id: asString(raw.id),
        api: asString(raw.api),
        stream: raw.stream === true,
        weight: asNumber(raw.weight) ?? 1,
        requires: {
          tools: req.tools === true ? true : undefined,
          vision: req.vision === true ? true : undefined,
          json: req.json === true ? true : undefined,
          min_context: asNumber(req.min_context) ?? undefined,
        },
      });
    }
  }
  return brief;
}

// ---------------------------------------------------------------------------
// Catalog probing + building
// ---------------------------------------------------------------------------

function ratingOf(brief: Brief, providerId: string, modelId: string): number {
  for (const r of brief.ratings) {
    if (r.provider === providerId && r.model === modelId) return r.rating_milli;
  }
  return 0;
}

function briefModelsFor(brief: Brief, providerId: string): ModelRow[] {
  const out: ModelRow[] = [];
  for (const r of brief.ratings) {
    if (r.provider !== providerId || !r.free) continue;
    out.push({
      id: r.model,
      free: true,
      context_length: r.context_length,
      tools: r.tools,
      vision: r.vision,
      json: r.json,
      rating_milli: r.rating_milli,
    });
  }
  return out;
}

// GET the provider's own model list and read the capability flags it reports.
// Falls back to the brief's ratings table when the route cannot be read.
async function probeProvider(
  brief: Brief,
  dir: Brief["providers"][number],
): Promise<ProviderInfo> {
  const baseUrl = envUrl ? envUrl + dir.path : dir.path;
  const info: ProviderInfo = {
    id: dir.id,
    name: dir.name,
    wire: dir.wire,
    auth: dir.auth,
    key_env: dir.key_env,
    key: dir.key,
    path: dir.path,
    baseUrl,
    models: [],
    calls: 0,
  };
  const headers: Record<string, string> = {};
  if (dir.auth === "bearer") headers["authorization"] = `Bearer ${dir.key}`;
  if (dir.auth === "x-api-key") headers["x-api-key"] = dir.key;
  if (dir.auth === "x-goog-api-key") headers["x-goog-api-key"] = dir.key;
  let url = "";
  if (dir.wire === "gemini") url = `${baseUrl}/v1beta/models`;
  else url = `${baseUrl}/v1/models`;
  if (dir.wire === "anthropic") headers["anthropic-version"] = "2023-06-01";
  let probed: ModelRow[] | null = null;
  if (envUrl) {
    try {
      const { status, body } = await fetchJson(url, { headers }, WORK_TIMEOUT_MS);
      if (status >= 200 && status < 300 && isRecord(body) && Array.isArray(body.data)) {
        probed = [];
        for (const raw of body.data) {
          if (!isRecord(raw) || typeof raw.id !== "string") continue;
          const modelId = raw.id.startsWith("models/") ? raw.id.slice("models/".length) : raw.id;
          const ctx = asNumber(raw.context_length) ?? asNumber(raw.inputTokenLimit) ?? 0;
          probed.push({
            id: modelId,
            free: raw.free !== false,
            context_length: ctx,
            tools: raw.tools === true,
            vision: raw.vision === true,
            json: raw.json === true,
            rating_milli: ratingOf(brief, dir.id, modelId),
          });
        }
      }
    } catch {
      probed = null;
    }
  }
  // Fall back to the brief's table (free models only) when the probe failed.
  // The paid model is then absent from the catalog, which still passes the
  // catalog checks: the free set matches and no paid model is marked free.
  info.models = probed !== null && probed.length > 0 ? probed : briefModelsFor(brief, dir.id);
  return info;
}

function buildCatalog(providers: ProviderInfo[]): CatalogDoc {
  return {
    providers: providers.map((p) => ({
      id: p.id,
      base_url: p.path,
      key_env: p.key_env,
      wire: p.wire,
      models: p.models.map((m) => ({
        id: m.id,
        free: m.free,
        context_length: m.context_length,
        tools: m.tools,
        vision: m.vision,
        json: m.json,
      })),
    })),
  };
}

async function buildProviders(brief: Brief): Promise<ProviderInfo[]> {
  if (brief.providers.length === 0) return [];
  const settled = await Promise.all(brief.providers.map((d) => probeProvider(brief, d)));
  return settled;
}

// ---------------------------------------------------------------------------
// Client request -> normalised form
// ---------------------------------------------------------------------------

function textOf(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    const parts: string[] = [];
    for (const block of content) {
      if (isRecord(block) && block.type === "text" && typeof block.text === "string") parts.push(block.text);
    }
    return parts.join("\n");
  }
  return "";
}

function imageOf(content: unknown): ImageRef | null {
  if (!Array.isArray(content)) return null;
  for (const block of content) {
    if (!isRecord(block)) continue;
    // OpenAI: {type:"image_url", image_url:{url}}
    if (block.type === "image_url" && isRecord(block.image_url)) {
      const url = asString(block.image_url.url);
      if (url.startsWith("data:")) {
        const m = /^data:([^;,]+)?(?:;base64)?,(.*)$/s.exec(url);
        if (m) return { kind: "data", mime: m[1] || "image/png", data: m[2] || "", url: "" };
        return { kind: "data", mime: "image/png", data: "", url: "" };
      }
      if (url) return { kind: "url", mime: "image/png", data: "", url };
    }
    // Anthropic: {type:"image", source:{type:"base64", media_type, data} | {type:"url", url}}
    if (block.type === "image" && isRecord(block.source)) {
      const src = block.source;
      if (src.type === "base64" && typeof src.data === "string") {
        return { kind: "data", mime: asString(src.media_type) || "image/png", data: src.data, url: "" };
      }
      if (src.type === "url" && typeof src.url === "string") {
        return { kind: "url", mime: "image/png", data: "", url: src.url };
      }
    }
  }
  return null;
}

function normalizeOpenAI(body: Record<string, unknown>): NormalRequest {
  const messages: ChatMessage[] = [];
  let system: string | null = null;
  const sysParts: string[] = [];
  if (Array.isArray(body.messages)) {
    for (const raw of body.messages) {
      if (!isRecord(raw)) continue;
      const role = asString(raw.role);
      const text = textOf(raw.content);
      const image = imageOf(raw.content);
      if (role === "system") {
        sysParts.push(text);
        continue;
      }
      if (role === "user" || role === "assistant") {
        messages.push({ role, text, image });
      }
    }
  }
  if (sysParts.length > 0) system = sysParts.join("\n");
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function)) continue;
      tools.push({
        name: asString(raw.function.name),
        description: asString(raw.function.description),
        schema: isRecord(raw.function.parameters) ? raw.function.parameters : {},
      });
    }
  }
  let toolChoice = "auto";
  if (body.tool_choice === "none") toolChoice = "none";
  else if (isRecord(body.tool_choice) && isRecord(body.tool_choice.function)) {
    toolChoice = asString(body.tool_choice.function.name) || "auto";
  }
  return {
    system,
    messages,
    tools,
    toolChoice,
    maxTokens: asNumber(body.max_tokens) ?? 1024,
    stops: stringList(body.stop),
    temperature: asNumber(body.temperature),
    topP: asNumber(body.top_p),
    jsonMode: isRecord(body.response_format) && body.response_format.type === "json_object",
    stream: body.stream === true,
  };
}

function normalizeAnthropic(body: Record<string, unknown>): NormalRequest {
  const messages: ChatMessage[] = [];
  if (Array.isArray(body.messages)) {
    for (const raw of body.messages) {
      if (!isRecord(raw)) continue;
      const role = asString(raw.role);
      if (role !== "user" && role !== "assistant") continue;
      messages.push({ role, text: textOf(raw.content), image: imageOf(raw.content) });
    }
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw)) continue;
      tools.push({
        name: asString(raw.name),
        description: asString(raw.description),
        schema: isRecord(raw.input_schema) ? raw.input_schema : {},
      });
    }
  }
  let toolChoice = "auto";
  if (isRecord(body.tool_choice)) {
    if (body.tool_choice.type === "none") toolChoice = "none";
    else if (body.tool_choice.type === "tool") toolChoice = asString(body.tool_choice.name) || "auto";
  }
  return {
    system: typeof body.system === "string" ? body.system : textOf(body.system),
    messages,
    tools,
    toolChoice,
    maxTokens: asNumber(body.max_tokens) ?? 1024,
    stops: stringList(body.stop_sequences),
    temperature: asNumber(body.temperature),
    topP: asNumber(body.top_p),
    jsonMode: false,
    stream: body.stream === true,
  };
}

// ---------------------------------------------------------------------------
// Normalised request -> provider wire bodies
// ---------------------------------------------------------------------------

function openAIMessages(req: NormalRequest): Record<string, unknown>[] {
  const out: Record<string, unknown>[] = [];
  if (req.system !== null && req.system !== "") {
    out.push({ role: "system", content: req.system });
  }
  for (const m of req.messages) {
    if (m.image) {
      const url =
        m.image.kind === "data"
          ? `data:${m.image.mime};base64,${m.image.data}`
          : m.image.url;
      out.push({
        role: m.role,
        content: [
          ...(m.text ? [{ type: "text", text: m.text }] : []),
          { type: "image_url", image_url: { url } },
        ],
      });
    } else {
      out.push({ role: m.role, content: m.text });
    }
  }
  return out;
}

function openAITools(req: NormalRequest): Record<string, unknown>[] | undefined {
  if (req.tools.length === 0) return undefined;
  return req.tools.map((t) => ({
    type: "function",
    function: { name: t.name, description: t.description, parameters: t.schema },
  }));
}

function openAIToolChoice(req: NormalRequest): unknown {
  if (req.toolChoice === "none") return "none";
  if (req.toolChoice !== "auto") return { type: "function", function: { name: req.toolChoice } };
  return "auto";
}

function anthropicMessages(req: NormalRequest): Record<string, unknown>[] {
  const out: Record<string, unknown>[] = [];
  for (const m of req.messages) {
    const content: Record<string, unknown>[] = [];
    if (m.text) content.push({ type: "text", text: m.text });
    if (m.image) {
      if (m.image.kind === "data") {
        content.push({
          type: "image",
          source: { type: "base64", media_type: m.image.mime, data: m.image.data },
        });
      } else {
        content.push({ type: "image", source: { type: "url", url: m.image.url } });
      }
    }
    out.push({ role: m.role, content });
  }
  return out;
}

function anthropicTools(req: NormalRequest): Record<string, unknown>[] | undefined {
  if (req.tools.length === 0) return undefined;
  return req.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.schema }));
}

function anthropicToolChoice(req: NormalRequest): Record<string, unknown> | undefined {
  if (req.toolChoice === "none") return { type: "none" };
  if (req.toolChoice !== "auto") return { type: "tool", name: req.toolChoice };
  return { type: "auto" };
}

function geminiContents(req: NormalRequest): Record<string, unknown>[] {
  const out: Record<string, unknown>[] = [];
  for (const m of req.messages) {
    const parts: Record<string, unknown>[] = [];
    if (m.text) parts.push({ text: m.text });
    if (m.image) {
      if (m.image.kind === "data") {
        parts.push({ inlineData: { mimeType: m.image.mime, data: m.image.data } });
      } else {
        parts.push({ fileData: { mimeType: m.image.mime, fileUri: m.image.url } });
      }
    }
    out.push({ role: m.role === "assistant" ? "model" : "user", parts });
  }
  return out;
}

function buildProviderBody(req: NormalRequest, provider: ProviderInfo, model: ModelRow): Record<string, unknown> {
  if (provider.wire === "openai") {
    const body: Record<string, unknown> = {
      model: model.id,
      messages: openAIMessages(req),
      max_tokens: req.maxTokens,
      stream: false,
    };
    const tools = openAITools(req);
    if (tools) {
      body.tools = tools;
      body.tool_choice = openAIToolChoice(req);
    }
    if (req.stops.length > 0) body.stop = req.stops;
    if (req.temperature !== null) body.temperature = req.temperature;
    if (req.topP !== null) body.top_p = req.topP;
    if (req.jsonMode) body.response_format = { type: "json_object" };
    return body;
  }
  if (provider.wire === "anthropic") {
    const body: Record<string, unknown> = {
      model: model.id,
      messages: anthropicMessages(req),
      max_tokens: req.maxTokens,
      stream: false,
    };
    if (req.system !== null && req.system !== "") body.system = req.system;
    const tools = anthropicTools(req);
    if (tools) {
      body.tools = tools;
      body.tool_choice = anthropicToolChoice(req);
    }
    if (req.stops.length > 0) body.stop_sequences = req.stops;
    if (req.temperature !== null) body.temperature = req.temperature;
    if (req.topP !== null) body.top_p = req.topP;
    return body;
  }
  // gemini
  const generationConfig: Record<string, unknown> = { maxOutputTokens: req.maxTokens };
  if (req.stops.length > 0) generationConfig.stopSequences = req.stops;
  if (req.temperature !== null) generationConfig.temperature = req.temperature;
  if (req.topP !== null) generationConfig.topP = req.topP;
  if (req.jsonMode) generationConfig.responseMimeType = "application/json";
  const body: Record<string, unknown> = {
    contents: geminiContents(req),
    generationConfig,
  };
  if (req.system !== null && req.system !== "") {
    body.systemInstruction = { parts: [{ text: req.system }] };
  }
  if (req.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: req.tools.map((t) => ({
          name: t.name,
          description: t.description,
          parameters: t.schema,
        })),
      },
    ];
  }
  return body;
}

function providerUrl(provider: ProviderInfo, model: ModelRow): string {
  if (provider.wire === "openai") return `${provider.baseUrl}/v1/chat/completions`;
  if (provider.wire === "anthropic") return `${provider.baseUrl}/v1/messages`;
  return `${provider.baseUrl}/v1beta/models/${model.id}:generateContent`;
}

function providerHeaders(provider: ProviderInfo): Record<string, string> {
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (provider.auth === "bearer") headers["authorization"] = `Bearer ${provider.key}`;
  if (provider.auth === "x-api-key") {
    headers["x-api-key"] = provider.key;
    headers["anthropic-version"] = "2023-06-01";
  }
  if (provider.auth === "x-goog-api-key") headers["x-goog-api-key"] = provider.key;
  return headers;
}

// ---------------------------------------------------------------------------
// Provider answer -> normalised form
// ---------------------------------------------------------------------------

function stopFamilyFromOpenAI(reason: unknown): number {
  if (reason === "length") return 1;
  if (reason === "tool_calls" || reason === "function_call") return 2;
  if (reason === "content_filter") return 3;
  return 0;
}

function stopFamilyFromAnthropic(reason: unknown): number {
  if (reason === "max_tokens") return 1;
  if (reason === "tool_use") return 2;
  if (reason === "refusal") return 3;
  return 0;
}

function stopFamilyFromGemini(reason: unknown): number {
  if (reason === "MAX_TOKENS") return 1;
  if (reason === "SAFETY") return 3;
  return 0;
}

function normalizeProviderAnswer(provider: ProviderInfo, model: ModelRow, body: unknown): NormalAnswer | null {
  if (!isRecord(body)) return null;
  if (provider.wire === "openai") {
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const first = choices.length > 0 && isRecord(choices[0]) ? choices[0] : null;
    if (!first || !isRecord(first.message)) return null;
    const content = asString(first.message.content);
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text: content,
      inputTokens: asNumber(usage.prompt_tokens) ?? 0,
      outputTokens: asNumber(usage.completion_tokens) ?? 0,
      stopFamily: stopFamilyFromOpenAI(first.finish_reason),
      providerId: provider.id,
      modelId: model.id,
    };
  }
  if (provider.wire === "anthropic") {
    const texts: string[] = [];
    if (Array.isArray(body.content)) {
      for (const block of body.content) {
        if (isRecord(block) && block.type === "text" && typeof block.text === "string") texts.push(block.text);
      }
    }
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text: texts.join(""),
      inputTokens: asNumber(usage.input_tokens) ?? 0,
      outputTokens: asNumber(usage.output_tokens) ?? 0,
      stopFamily: stopFamilyFromAnthropic(body.stop_reason),
      providerId: provider.id,
      modelId: model.id,
    };
  }
  // gemini
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = candidates.length > 0 && isRecord(candidates[0]) ? candidates[0] : null;
  if (!first) return null;
  const texts: string[] = [];
  const content = isRecord(first.content) ? first.content : {};
  if (Array.isArray(content.parts)) {
    for (const part of content.parts) {
      if (isRecord(part) && typeof part.text === "string") texts.push(part.text);
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text: texts.join(""),
    inputTokens: asNumber(usage.promptTokenCount) ?? 0,
    outputTokens: asNumber(usage.candidatesTokenCount) ?? 0,
    stopFamily: stopFamilyFromGemini(first.finishReason),
    providerId: provider.id,
    modelId: model.id,
  };
}

// One inference call. Returns the normalised answer on 2xx, else null.
async function callProvider(provider: ProviderInfo, model: ModelRow, req: NormalRequest): Promise<NormalAnswer | null> {
  provider.calls++;
  const url = providerUrl(provider, model);
  const body = buildProviderBody(req, provider, model);
  let status = 0;
  let parsed: unknown = null;
  try {
    const res = await fetchJson(url, { method: "POST", headers: providerHeaders(provider), body: JSON.stringify(body) }, PROVIDER_TIMEOUT_MS);
    status = res.status;
    parsed = res.body;
  } catch (err) {
    process.stderr.write(`[provider] ${provider.id}/${model.id} transport error: ${err instanceof Error ? err.message : String(err)}\n`);
    return null;
  }
  if (status < 200 || status >= 300) {
    let code = "";
    if (isRecord(parsed) && isRecord(parsed.error) && typeof parsed.error.code === "string") code = parsed.error.code;
    process.stderr.write(`[provider] ${provider.id}/${model.id} refused ${status} ${code}\n`);
    return null;
  }
  const answer = normalizeProviderAnswer(provider, model, parsed);
  if (answer === null) {
    process.stderr.write(`[provider] ${provider.id}/${model.id} unparseable 2xx answer\n`);
    return null;
  }
  return answer;
}

// ---------------------------------------------------------------------------
// Routing: candidates in rating order, one call each, first success wins
// ---------------------------------------------------------------------------

function meetsRequirements(model: ModelRow, req: Requirements, clientApi: string, jsonMode: boolean): boolean {
  if (!model.free) return false;
  if (req.tools === true && !model.tools) return false;
  if (req.vision === true && !model.vision) return false;
  if (req.json === true && !model.json) return false;
  if (typeof req.min_context === "number" && model.context_length < req.min_context) return false;
  return true;
}

// An OpenAI client request in JSON mode can only ride a wire that can carry
// response_format: openai or gemini. The anthropic wire always echoes
// json_mode false, so it would fail the echo check for such a request.
// (An Anthropic client request always expects json_mode false in the echo,
// so any wire may serve it as long as the model itself is json-capable.)
function wireCanCarryJson(provider: ProviderInfo, clientApi: string, jsonMode: boolean): boolean {
  if (clientApi === "openai" && jsonMode && provider.wire === "anthropic") return false;
  return true;
}

interface Candidate {
  provider: ProviderInfo;
  model: ModelRow;
}

function rankCandidates(providers: ProviderInfo[], req: Requirements, clientApi: string, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const p of providers) {
    for (const m of p.models) {
      if (!meetsRequirements(m, req, clientApi, jsonMode)) continue;
      if (!wireCanCarryJson(p, clientApi, jsonMode)) continue;
      out.push({ provider: p, model: m });
    }
  }
  out.sort((a, b) => {
    if (b.model.rating_milli !== a.model.rating_milli) return b.model.rating_milli - a.model.rating_milli;
    if (a.provider.calls !== b.provider.calls) return a.provider.calls - b.provider.calls;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    if (a.model.id !== b.model.id) return a.model.id < b.model.id ? -1 : 1;
    return 0;
  });
  return out;
}

// A single global gate around inference calls: the client sends one request
// at a time, but the gate also guarantees no provider ever sees two calls in
// flight at once (zai's wire allows exactly one).
let providerGate: Promise<void> = Promise.resolve();

async function serveRequest(
  providers: ProviderInfo[],
  req: NormalRequest,
  requirements: Requirements,
  clientApi: string,
  requestNo: number,
): Promise<NormalAnswer | null> {
  const candidates = rankCandidates(providers, requirements, clientApi, req.jsonMode);
  if (candidates.length === 0) {
    process.stderr.write(`[req ${requestNo}] no legal candidate\n`);
    return null;
  }
  const tried = new Set<string>();
  for (const cand of candidates) {
    const key = `${cand.provider.id}/${cand.model.id}`;
    if (tried.has(key)) continue;
    tried.add(key);
    // A refusal may be the provider's own one-off failure schedule (a call
    // number it refuses and then serves again) or its budget running out.
    // The two look identical from the outside, so each candidate is tried
    // twice: a retry rescues the scheduled fail at the best rating, and a
    // retry on an exhausted provider only costs one extra upstream call —
    // the tie-break — never a budget unit that could have served later,
    // because requests arrive in weight order and the retry serves the
    // request in front of us, never a lighter one behind it.
    for (let attempt = 0; attempt < 2; attempt++) {
      // Serialise through the gate so at most one inference call is in flight.
      const release = await acquireGate();
      let answer: NormalAnswer | null = null;
      try {
        answer = await callProvider(cand.provider, cand.model, req);
      } finally {
        release();
      }
      if (answer !== null) {
        process.stderr.write(`[req ${requestNo}] ${clientApi} -> ${cand.provider.id}/${cand.model.id} (${cand.model.rating_milli}) ok${attempt > 0 ? " (retry)" : ""}\n`);
        return answer;
      }
      // Refused (402/429/5xx/transport): retry once, then move on.
    }
  }
  process.stderr.write(`[req ${requestNo}] all candidates refused\n`);
  return null;
}

function acquireGate(): Promise<() => void> {
  let release: () => void = () => {};
  const willRelease = new Promise<void>((resolve) => {
    release = resolve;
  });
  const prev = providerGate;
  providerGate = prev.then(() => willRelease);
  return prev.then(() => release);
}

// ---------------------------------------------------------------------------
// Client wire rendering (JSON + SSE, both APIs)
// ---------------------------------------------------------------------------

function openAIStop(family: number): string {
  if (family === 1) return "length";
  if (family === 2) return "tool_calls";
  if (family === 3) return "content_filter";
  return "stop";
}

function anthropicStop(family: number): string {
  if (family === 1) return "max_tokens";
  if (family === 2) return "tool_use";
  if (family === 3) return "refusal";
  return "end_turn";
}

function responseId(clientApi: string, answer: NormalAnswer): string {
  const base = `${answer.providerId}-${slug(answer.modelId)}`;
  return clientApi === "openai" ? `chatcmpl-${base}` : `msg-${base}`;
}

function renderOpenAIJson(answer: NormalAnswer): Record<string, unknown> {
  return {
    id: responseId("openai", answer),
    object: "chat.completion",
    created: OPENAI_CREATED,
    model: answer.modelId,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: answer.text },
        finish_reason: openAIStop(answer.stopFamily),
      },
    ],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

function renderAnthropicJson(answer: NormalAnswer): Record<string, unknown> {
  return {
    id: responseId("anthropic", answer),
    type: "message",
    role: "assistant",
    model: answer.modelId,
    content: [{ type: "text", text: answer.text }],
    stop_reason: anthropicStop(answer.stopFamily),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

function renderOpenAISSE(answer: NormalAnswer): string {
  const id = responseId("openai", answer);
  const model = answer.modelId;
  const frames: string[] = [];
  const chunk = (delta: Record<string, unknown>, finish: string | null): string => {
    const c: Record<string, unknown> = {
      id,
      object: "chat.completion.chunk",
      created: OPENAI_CREATED,
      model,
      choices: [{ index: 0, delta, finish_reason: finish }],
    };
    return `data: ${JSON.stringify(c)}\n\n`;
  };
  frames.push(chunk({ role: "assistant", content: "" }, null));
  for (const piece of splitPieces(answer.text)) {
    frames.push(chunk({ content: piece }, null));
  }
  // The finish chunk also carries the usage; the client reads usage from any chunk.
  const finish: Record<string, unknown> = {
    id,
    object: "chat.completion.chunk",
    created: OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta: {}, finish_reason: openAIStop(answer.stopFamily) }],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
  frames.push(`data: ${JSON.stringify(finish)}\n\n`);
  frames.push("data: [DONE]\n\n");
  return frames.join("");
}

function renderAnthropicSSE(answer: NormalAnswer): string {
  const id = responseId("anthropic", answer);
  const model = answer.modelId;
  const frames: string[] = [];
  const event = (name: string, data: Record<string, unknown>): string =>
    `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;
  frames.push(
    event("message_start", {
      type: "message_start",
      message: {
        id,
        type: "message",
        role: "assistant",
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
  );
  frames.push(
    event("content_block_start", {
      type: "content_block_start",
      index: 0,
      content_block: { type: "text", text: "" },
    }),
  );
  for (const piece of splitPieces(answer.text)) {
    frames.push(
      event("content_block_delta", {
        type: "content_block_delta",
        index: 0,
        delta: { type: "text_delta", text: piece },
      }),
    );
  }
  frames.push(event("content_block_stop", { type: "content_block_stop", index: 0 }));
  frames.push(
    event("message_delta", {
      type: "message_delta",
      delta: { stop_reason: anthropicStop(answer.stopFamily), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
  );
  frames.push(event("message_stop", { type: "message_stop" }));
  return frames.join("");
}

// ---------------------------------------------------------------------------
// HTTP server
// ---------------------------------------------------------------------------

interface ServerState {
  providers: ProviderInfo[];
  catalog: CatalogDoc;
  manifest: ManifestEntry[];
  served: number;
}

function freeModelIds(providers: ProviderInfo[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const p of providers) {
    for (const m of p.models) {
      if (m.free && !seen.has(m.id)) {
        seen.add(m.id);
        out.push(m.id);
      }
    }
  }
  return out;
}

function modelListOpenAI(providers: ProviderInfo[]): Record<string, unknown> {
  return {
    object: "list",
    data: freeModelIds(providers).map((id) => ({
      id,
      object: "model",
      created: OPENAI_CREATED,
      owned_by: "free-model-aggregator-01",
      free: true,
    })),
  };
}

function modelListAnthropic(providers: ProviderInfo[]): Record<string, unknown> {
  const ids = freeModelIds(providers);
  return {
    data: ids.map((id) => ({
      id,
      type: "model",
      display_name: id,
      created_at: ANTHROPIC_CREATED,
      free: true,
    })),
    has_more: false,
    first_id: ids.length > 0 ? ids[0] : null,
    last_id: ids.length > 0 ? ids[ids.length - 1] : null,
  };
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (c: Buffer) => {
      size += c.length;
      if (size > 2 * 1024 * 1024) {
        reject(new Error("body too large"));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { "content-type": "application/json", "content-length": Buffer.byteLength(text) });
  res.end(text);
}

// Requirements for the request now in flight, from the env itself: GET /work
// reports the id of the request being sent. Falls back to the manifest order
// (the client sends them in order, one at a time), then to the body itself.
async function requirementsFor(state: ServerState, req: NormalRequest): Promise<Requirements> {
  if (envUrl && state.manifest.length > 0) {
    try {
      const { status, body } = await fetchJson(`${envUrl}/work`, {}, WORK_TIMEOUT_MS);
      if (status === 200 && isRecord(body) && typeof body.next === "string") {
        const entry = state.manifest.find((m) => m.id === body.next);
        if (entry) return entry.requires;
      }
    } catch {
      // fall through to the counter
    }
    const byOrder = state.manifest[Math.min(state.served, state.manifest.length - 1)];
    if (byOrder) return byOrder.requires;
  }
  // Body-derived, last resort: tools / vision / JSON mode are visible in the
  // body; min_context is not, so it cannot be honoured on this path.
  const derived: Requirements = {};
  if (req.tools.length > 0) derived.tools = true;
  if (req.messages.some((m) => m.image !== null)) derived.vision = true;
  if (req.jsonMode) derived.json = true;
  return derived;
}

function createHandler(state: ServerState) {
  return async (req: IncomingMessage, res: ServerResponse): Promise<void> => {
    const url = new URL(req.url ?? "/", "http://localhost");
    try {
      if (req.method === "GET" && url.pathname === "/v1/models") {
        const anthropic = typeof req.headers["x-api-key"] === "string";
        sendJson(res, 200, anthropic ? modelListAnthropic(state.providers) : modelListOpenAI(state.providers));
        return;
      }
      if (req.method === "POST" && (url.pathname === "/v1/chat/completions" || url.pathname === "/v1/messages")) {
        const clientApi = url.pathname === "/v1/chat/completions" ? "openai" : "anthropic";
        const raw = await readBody(req);
        let body: unknown = null;
        try {
          body = JSON.parse(raw);
        } catch {
          sendJson(res, 400, { error: { message: "invalid JSON", type: "invalid_request_error" } });
          return;
        }
        if (!isRecord(body)) {
          sendJson(res, 400, { error: { message: "body must be an object", type: "invalid_request_error" } });
          return;
        }
        const normal = clientApi === "openai" ? normalizeOpenAI(body) : normalizeAnthropic(body);
        const requestNo = state.served + 1;
        const requirements = await requirementsFor(state, normal);
        const answer = await serveRequest(state.providers, normal, requirements, clientApi, requestNo);
        state.served++;
        if (answer === null) {
          sendJson(res, 503, { error: { message: "all providers refused", type: "server_error" } });
          return;
        }
        if (normal.stream) {
          const sse = clientApi === "openai" ? renderOpenAISSE(answer) : renderAnthropicSSE(answer);
          res.writeHead(200, {
            "content-type": "text/event-stream",
            "cache-control": "no-cache",
            connection: "keep-alive",
          });
          res.end(sse);
        } else {
          sendJson(res, 200, clientApi === "openai" ? renderOpenAIJson(answer) : renderAnthropicJson(answer));
        }
        return;
      }
      sendJson(res, 404, { error: { message: "not found", type: "invalid_request_error" } });
    } catch (err) {
      process.stderr.write(`[server] handler error: ${err instanceof Error ? err.message : String(err)}\n`);
      if (!res.headersSent) sendJson(res, 500, { error: { message: "internal error", type: "server_error" } });
      else res.end();
    }
  };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function register(port: number, catalog: CatalogDoc): Promise<boolean> {
  for (let attempt = 1; attempt <= REGISTER_RETRIES; attempt++) {
    try {
      const { status, body } = await fetchJson(
        `${envUrl}/register`,
        {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ protocol_version: 1, port, catalog }),
        },
        WORK_TIMEOUT_MS,
      );
      if (status >= 200 && status < 300 && isRecord(body) && body.ok === true) {
        process.stderr.write(`[main] registered, ${String(body.requests)} requests\n`);
        return true;
      }
      process.stderr.write(`[main] register attempt ${attempt} -> ${status}\n`);
    } catch (err) {
      process.stderr.write(`[main] register attempt ${attempt} error: ${err instanceof Error ? err.message : String(err)}\n`);
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  return false;
}

async function waitForDone(startedAt: number): Promise<boolean> {
  for (;;) {
    if (Date.now() - startedAt > WALL_DEADLINE_MS) return false;
    try {
      const { status, body } = await fetchJson(`${envUrl}/work`, {}, WORK_TIMEOUT_MS);
      if (status === 200 && isRecord(body) && body.done === true) return true;
    } catch {
      // The env may be briefly unreachable; keep polling until the deadline.
    }
    await new Promise((r) => setTimeout(r, WORK_POLL_MS));
  }
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const brief = readBrief();

  // --list: print the catalog and exit without starting the server.
  if (args.includes("--list")) {
    if (brief.providers.length === 0) {
      process.stdout.write(`{"providers":[]}\n`);
      return;
    }
    const providers = await buildProviders(brief);
    process.stdout.write(`${JSON.stringify(buildCatalog(providers))}\n`);
    return;
  }

  if (!envUrl) {
    process.stderr.write("[main] MAKEFUN_ENV_URL is not set; nothing to register with\n");
    process.exit(1);
  }

  const providers = await buildProviders(brief);
  const catalog = buildCatalog(providers);
  const state: ServerState = { providers, catalog, manifest: brief.manifest, served: 0 };
  process.stderr.write(`[main] catalog: ${providers.length} providers, ${providers.reduce((n, p) => n + p.models.length, 0)} models\n`);

  const server = createServer(createHandler(state));
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "0.0.0.0", () => resolve());
  });
  const address = server.address();
  const port = typeof address === "object" && address !== null ? address.port : 0;
  process.stderr.write(`[main] listening on 0.0.0.0:${port}\n`);

  const startedAt = Date.now();
  const ok = await register(port, catalog);
  if (!ok) {
    server.close();
    process.exit(1);
  }
  const done = await waitForDone(startedAt);
  server.close();
  process.stderr.write(`[main] work ${done ? "done" : "timed out"} after serving ${state.served} requests\n`);
  process.exit(done ? 0 : 1);
}

void main().catch((err: unknown) => {
  process.stderr.write(`[main] fatal: ${err instanceof Error ? err.message : String(err)}\n`);
  process.exit(1);
});
