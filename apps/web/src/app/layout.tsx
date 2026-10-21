import type { Metadata, Viewport } from "next";
import { Atkinson_Hyperlegible, Bricolage_Grotesque } from "next/font/google";
import { ScanProvider } from "@/components/scan-provider";

import "./globals.css";

const bricolage = Bricolage_Grotesque({ subsets: ["latin"], variable: "--font-bricolage" });
const atkinson = Atkinson_Hyperlegible({
  subsets: ["latin"],
  weight: ["400", "700"],
  variable: "--font-atkinson",
});

export const metadata: Metadata = {
  title: "Bin Buddy",
  description: "Which bin does this go in? An explainable recycling assistant for Singapore.",
};

export const viewport: Viewport = { themeColor: "#10233F" };

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en-SG" className={`h-full antialiased ${bricolage.variable} ${atkinson.variable}`}>
      <body className="min-h-full flex flex-col">
        {/* Mounted once for the whole app, so a scan survives switching tabs. */}
        <ScanProvider>{children}</ScanProvider>
      </body>
    </html>
  );
}
