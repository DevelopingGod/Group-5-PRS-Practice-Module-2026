// Browser-only: shrinks a photo before upload so it stays well under MAX_IMAGE_BYTES
// and uploads quickly on mobile data. Re-encoding through a canvas also drops EXIF metadata.
import { MAX_IMAGE_BYTES } from "@/contracts/constants";

const MAX_SIDE = 1024;
// The copy kept in sessionStorage so a reload can still show the item: small, no EXIF.
const THUMBNAIL_SIDE = 320;

async function drawScaled(file: File, maxSide: number): Promise<HTMLCanvasElement> {
  const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas is not available");
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();
  return canvas;
}

/** Returns raw base64 JPEG (no `data:` prefix), as the model API expects. */
export async function fileToBase64Jpeg(file: File): Promise<string> {
  const canvas = await drawScaled(file, MAX_SIDE);
  for (const quality of [0.85, 0.7, 0.5]) {
    const base64 = canvas.toDataURL("image/jpeg", quality).split(",", 2)[1] ?? "";
    if ((base64.length * 3) / 4 <= MAX_IMAGE_BYTES) return base64;
  }
  throw new Error("Photo is too large");
}

/** A small `data:image/jpeg` URL of the photo (about 10–20 kB). */
export async function fileToThumbnail(file: File): Promise<string> {
  const canvas = await drawScaled(file, THUMBNAIL_SIDE);
  return canvas.toDataURL("image/jpeg", 0.7);
}
