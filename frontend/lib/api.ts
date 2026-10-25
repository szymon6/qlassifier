// HTTP client for the local API (TDD §6). Every failure becomes an ApiError whose
// message + hint are exactly the server's text (US-5); network failures become
// "Server stopped." (TDD §6.5).
import type { ApiErrorBody, BenchmarkResult, DatasetCard, Meta, Run } from "./types";

export const API_BASE: string = (process.env.NEXT_PUBLIC_API_BASE ?? "").replace(/\/$/, "");
export const API_PREFIX = "/api/v1";

export class ApiError extends Error {
  readonly code: string;
  readonly hint: string;
  readonly status: number;

  constructor(body: ApiErrorBody) {
    super(body.message);
    this.name = "ApiError";
    this.code = body.code;
    this.hint = body.hint;
    this.status = body.status;
  }

  get body(): ApiErrorBody {
    return { code: this.code, message: this.message, hint: this.hint, status: this.status };
  }
}

export const SERVER_STOPPED: ApiErrorBody = {
  code: "server_stopped",
  message: "Server stopped.",
  hint: "Run `qlassifier ui` again.",
  status: 0,
};

export const VERSION_MISMATCH: ApiErrorBody = {
  code: "version_mismatch",
  message: "Dashboard and library versions differ.",
  hint: "Reinstall qlassifier.",
  status: 0,
};

export function isErrorBody(value: unknown): value is ApiErrorBody {
  if (!value || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  return typeof v.code === "string" && typeof v.message === "string" && typeof v.hint === "string";
}

/** Extracts `{error: {...}}` from a response body, or synthesises one from the status. */
export async function errorFromResponse(response: Response): Promise<ApiError> {
  try {
    const payload: unknown = await response.json();
    const error = (payload as { error?: unknown } | null)?.error;
    if (isErrorBody(error))
      return new ApiError({ ...error, status: error.status ?? response.status });
  } catch {
    // Body is not JSON: fall through to a status-only error.
  }
  return new ApiError({
    code: "http_error",
    message: `Request failed (${response.status}).`,
    hint: "See the terminal running `qlassifier ui`.",
    status: response.status,
  });
}

type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;

export interface ApiClient {
  meta(): Promise<Meta>;
  datasets(): Promise<DatasetCard[]>;
  result(dataset: string): Promise<BenchmarkResult>;
  startRun(dataset: string, seeds?: number): Promise<Run>;
  run(runId: string): Promise<Run>;
  runResult(runId: string): Promise<BenchmarkResult>;
  cancelRun(runId: string): Promise<Run>;
  eventsUrl(runId: string): string;
}

export function createApi(base: string = API_BASE, fetchImpl?: FetchLike): ApiClient {
  const doFetch: FetchLike = fetchImpl ?? ((input, init) => fetch(input, init));
  const url = (path: string) => `${base}${API_PREFIX}${path}`;

  async function request<T>(path: string, init?: RequestInit): Promise<T> {
    let response: Response;
    try {
      response = await doFetch(url(path), {
        ...init,
        headers: { Accept: "application/json", ...(init?.headers ?? {}) },
      });
    } catch {
      throw new ApiError(SERVER_STOPPED);
    }
    if (!response.ok) throw await errorFromResponse(response);
    return (await response.json()) as T;
  }

  const post = <T>(path: string, body?: unknown) =>
    request<T>(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body ?? {}),
    });

  return {
    meta: () => request<Meta>("/meta"),
    datasets: async () => (await request<{ datasets: DatasetCard[] }>("/datasets")).datasets,
    result: (dataset) => request<BenchmarkResult>(`/results/${encodeURIComponent(dataset)}`),
    startRun: (dataset, seeds) => post<Run>("/runs", seeds ? { dataset, seeds } : { dataset }),
    run: (runId) => request<Run>(`/runs/${encodeURIComponent(runId)}`),
    runResult: (runId) => request<BenchmarkResult>(`/runs/${encodeURIComponent(runId)}/result`),
    cancelRun: (runId) => post<Run>(`/runs/${encodeURIComponent(runId)}/cancel`),
    eventsUrl: (runId) => url(`/runs/${encodeURIComponent(runId)}/events`),
  };
}

/** The UI speaks API v1, schema major 1 (TDD §6.1). */
export function isCompatible(meta: Pick<Meta, "api_version" | "schema_version">): boolean {
  return meta.api_version === "1" && String(meta.schema_version).split(".")[0] === "1";
}
