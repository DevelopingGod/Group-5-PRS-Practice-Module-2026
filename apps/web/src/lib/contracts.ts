// Friendly names for the generated contract types. Never hand-write API shapes:
// they come from contracts/generated/openapi.json via `pnpm gen:types`.
import type { components } from "@/contracts/api";

type Schemas = components["schemas"];

export type PredictRequest = Schemas["PredictRequest"];
export type PredictResponse = Schemas["PredictResponse"];
export type AnswerRequest = Schemas["AnswerRequest"];
export type AnswerResponse = Schemas["AnswerResponse"];
export type VerifyRequest = Schemas["VerifyRequest"];
export type VerifyResponse = Schemas["VerifyResponse"];
export type HealthResponse = Schemas["HealthResponse"];
export type VersionResponse = Schemas["VersionResponse"];
export type ErrorResponse = Schemas["ErrorResponse"];
export type VerdictResult = Schemas["VerdictResult"];
export type Question = Schemas["Question"];
export type DecisionState = Schemas["DecisionState"];
export type BinQrPayload = Schemas["BinQrPayload"];
export type Material = Schemas["Material"];
export type Condition = Schemas["Condition"];
export type Channel = Schemas["Channel"];
