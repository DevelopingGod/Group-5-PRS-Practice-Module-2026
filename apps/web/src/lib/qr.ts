import Ajv2020 from "ajv/dist/2020";

import schema from "@/contracts/bin-qr.schema.json";
import type { BinQrPayload } from "@/lib/contracts";

const validate = new Ajv2020({ allErrors: true }).compile<BinQrPayload>(schema);

export type QrResult = { ok: true; payload: BinQrPayload } | { ok: false; error: string };

/** Parse the text read from a bin's QR code and check it against the contract. */
export function parseBinQr(text: string): QrResult {
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    return { ok: false, error: "QR code does not contain JSON" };
  }
  if (!validate(data)) {
    const detail = validate.errors?.map((e) => `${e.instancePath || "/"} ${e.message}`).join("; ");
    return { ok: false, error: `Not a Bin Buddy bin QR code: ${detail}` };
  }
  return { ok: true, payload: data };
}
