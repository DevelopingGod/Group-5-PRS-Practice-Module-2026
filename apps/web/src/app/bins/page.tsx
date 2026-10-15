import type { Metadata } from "next";
import QRCode from "qrcode";

import { DEMO_BINS, encodeBinQr } from "@/lib/demo-bins";

// Printable sheet of QR codes for the demonstration bins. Not linked from the app.
export const metadata: Metadata = {
  title: "Demo bin QR codes · Bin Buddy",
  robots: { index: false, follow: false },
};

export default async function BinsPage() {
  const codes = await Promise.all(
    DEMO_BINS.map(async (bin) => ({
      bin,
      svg: await QRCode.toString(encodeBinQr(bin), {
        type: "svg",
        errorCorrectionLevel: "M",
        margin: 2,
        color: { dark: "#10233F", light: "#FFFFFF" },
      }),
    })),
  );

  return (
    <main className="mx-auto w-full max-w-4xl px-4 py-6 print:p-0">
      <h1 className="font-heading text-2xl font-bold print:hidden">Demo bin QR codes</h1>
      <p className="mt-1 text-muted print:hidden">
        Print this page and stick each code on its demonstration bin. Demonstration bins only:
        public HDB bins need town council approval.
      </p>
      <ul className="mt-6 grid grid-cols-1 gap-6 sm:grid-cols-2 print:grid-cols-2">
        {codes.map(({ bin, svg }) => (
          <li
            key={bin.bin_id}
            className="break-inside-avoid rounded-2xl border-2 border-ink bg-white p-5 text-center"
          >
            <p className="font-heading text-xl font-bold">Bin Buddy</p>
            <div
              role="img"
              aria-label={`QR code for ${bin.location_label}`}
              className="mx-auto my-3 w-56"
              dangerouslySetInnerHTML={{ __html: svg }}
            />
            <p className="font-bold">{bin.location_label}</p>
            <p className="text-sm text-muted">
              {bin.bin_id} · {bin.site_policy} · {bin.bin_key}
            </p>
          </li>
        ))}
      </ul>
    </main>
  );
}
